"""Train the V4 hybrid room model using full open-loop rollouts."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import (
    assign_v4_splits, compute_normalization, describe_runs,
    discover_v4_cooling_runs, run_arrays,
)
from .model import CONTROL_COLUMNS, INITIAL_COLUMNS, HybridRoomV4, V4Config


def make_padded_batch(runs: list[dict], initial_median: np.ndarray, device: torch.device):
    arrays = [run_arrays(run, initial_median) for run in runs]
    lengths = torch.tensor([len(item[1]) for item in arrays], device=device)
    max_length = int(lengths.max().item())
    initial = torch.tensor(np.stack([item[0] for item in arrays]), device=device)
    controls = torch.zeros((len(runs), max_length, 3), device=device)
    truth = torch.zeros((len(runs), max_length + 1), device=device)
    mask = torch.zeros((len(runs), max_length + 1), dtype=torch.bool, device=device)
    for index, (_, run_controls, run_truth) in enumerate(arrays):
        n = len(run_controls)
        controls[index, :n] = torch.tensor(run_controls, device=device)
        truth[index, :n + 1] = torch.tensor(run_truth[:n + 1], device=device)
        mask[index, :n + 1] = True
    return initial, controls, truth, mask, lengths


def masked_loss(prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = prediction[mask] - truth[mask]
    # MAE drives long-horizon accuracy; the quadratic term still discourages
    # large run-level drift without letting a few quantized steps dominate.
    base = error.abs().mean() + 0.20 * torch.sqrt((error.square()).mean() + 1e-8)
    late_errors = []
    end_errors = []
    dynamic_errors = []
    amplitude_errors = []
    for row in range(prediction.shape[0]):
        length = int(mask[row].sum().item())
        late_start = max(1, int(length * 0.75))
        late_errors.append((prediction[row, late_start:length] - truth[row, late_start:length]).abs().mean())
        end_errors.append((prediction[row, length - 1] - truth[row, length - 1]).abs())
        for horizon, horizon_weight in ((12, 0.35), (60, 0.65)):
            if length <= horizon:
                continue
            predicted_change = prediction[row, horizon:length] - prediction[row, :length - horizon]
            measured_change = truth[row, horizon:length] - truth[row, :length - horizon]
            measured_scale = measured_change.std().detach().clamp_min(
                0.03 if horizon == 12 else 0.10
            )
            dynamic_errors.append(
                horizon_weight
                * (predicted_change - measured_change).abs().mean()
                / measured_scale
            )
            amplitude_errors.append(
                horizon_weight
                * (predicted_change.std() / measured_scale - 1.0).abs()
            )
    dynamic_loss = torch.stack(dynamic_errors).mean()
    amplitude_loss = torch.stack(amplitude_errors).mean()
    return (
        base
        + 0.15 * torch.stack(late_errors).mean()
        + 0.15 * torch.stack(end_errors).mean()
        + 0.12 * dynamic_loss
        + 0.08 * amplitude_loss
    )


def capacity_pretraining_loss(
    model: HybridRoomV4,
    initial: torch.Tensor,
    controls: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    horizon: int = 12,
) -> torch.Tensor:
    """Teacher-forced 60 s supervision for normalized sensible capacity.

    The target follows directly from the normalized Appendix-C heat balance.
    A central standard load coefficient is used only for pretraining; the
    subsequent open-loop objective learns the initialization-dependent value.
    """
    state = model.initialize(initial)
    capacities = []
    for index in range(controls.shape[1]):
        # Teacher forcing is restricted to capacity pretraining.  The final
        # training and every evaluation remain strictly open loop.
        state["temperature"] = truth[:, index]
        state = model.step_state(state, controls[:, index])
        capacities.append(state.get("effective_cooling", state["cooling"]))
    predicted = torch.stack(capacities, dim=1)

    target_length = truth.shape[1] - horizon
    current_temperature = truth[:, :target_length]
    future_temperature = truth[:, horizon:horizon + target_length]
    outdoor = initial[:, INITIAL_COLUMNS.index("T_out")].unsqueeze(1)
    load = 0.053 * torch.clamp(outdoor - current_temperature + 4.0, min=0.0)
    temperature_rate = 125.0 * (
        future_temperature - current_temperature
    ) / (horizon * 5.0)
    target = torch.clamp(load - temperature_rate, min=0.0, max=2.0)

    valid = mask[:, horizon:horizon + target_length].clone()
    # The virtual-load loop begins only after compressor startup.  Excluding
    # the pre-start chamber-conditioning samples prevents a false capacity
    # target while measured room temperature is held constant externally.
    has_started = torch.cumsum((controls[:, :target_length, 0] > 0.5).to(torch.int32), dim=1) > 0
    valid &= has_started
    error = predicted[:, :target_length][valid] - target[valid]
    return torch.nn.functional.smooth_l1_loss(
        error, torch.zeros_like(error), beta=0.10,
    )


def residual_pretraining_loss(
    model: HybridRoomV4,
    initial: torch.Tensor,
    controls: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    horizon: int = 12,
) -> torch.Tensor:
    """Directly supervise fast residual room-temperature rate at 60 s scale."""
    state = model.initialize(initial)
    predicted_rates = []
    physical_rates = []
    for index in range(controls.shape[1]):
        state["temperature"] = truth[:, index]
        state = model.step_state(state, controls[:, index])
        predicted_rates.append(state["last_residual_rate_c_per_min"])
        physical_rates.append(
            60.0 / 125.0 * (state["last_load"] - state["effective_cooling"])
        )
    predicted = torch.stack(predicted_rates, dim=1)
    physical = torch.stack(physical_rates, dim=1).detach()

    target_length = truth.shape[1] - horizon
    measured_rate = 60.0 * (
        truth[:, horizon:horizon + target_length] - truth[:, :target_length]
    ) / (horizon * 5.0)
    limit = model.config.residual_rate_max_c_per_min
    target = torch.clamp(
        measured_rate - physical[:, :target_length], min=-limit, max=limit,
    )
    valid = mask[:, horizon:horizon + target_length].clone()
    has_started = torch.cumsum(
        (controls[:, :target_length, 0] > 0.5).to(torch.int32), dim=1,
    ) > 0
    valid &= has_started
    return torch.nn.functional.smooth_l1_loss(
        predicted[:, :target_length][valid], target[valid], beta=0.05,
    )


def save_payload(path: Path, model: HybridRoomV4, normalization: dict, metadata: dict) -> None:
    payload = {
        "format": "haier_room_v4_torch_state_dict",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": model.config.to_dict(),
        "initial_columns": INITIAL_COLUMNS,
        "control_columns": CONTROL_COLUMNS,
        "normalization": {key: value.tolist() for key, value in normalization.items()},
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset_full"))
    parser.add_argument("--output", type=Path,
                        default=Path("simu/roomv4/room_v4_model.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--capacity-pretrain-epochs", type=int, default=25)
    parser.add_argument("--residual-pretrain-epochs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--tau-max-seconds", type=float, default=600.0)
    parser.add_argument("--fast-capacity-fraction", type=float, default=0.0)
    parser.add_argument("--capacity-aux-weight", type=float, default=0.0)
    parser.add_argument("--max-temp-rate", type=float, default=0.6)
    parser.add_argument("--residual-rate-max", type=float, default=0.0)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume/fine-tune a compatible V4 state_dict payload.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    runs = discover_v4_cooling_runs(args.data_dir)
    assign_v4_splits(runs)
    training_runs = [run for run in runs if run["split"] == "train"]
    normalization = compute_normalization(training_runs)
    initial, controls, truth, mask, _ = make_padded_batch(
        training_runs, normalization["initial_mean"], device,
    )
    config = V4Config(
        hidden_size=args.hidden_size,
        cooling_tau_max_seconds=args.tau_max_seconds,
        fast_capacity_fraction=args.fast_capacity_fraction,
        max_temperature_rate_c_per_min=args.max_temp_rate,
        residual_rate_max_c_per_min=args.residual_rate_max,
    )
    model = HybridRoomV4(**normalization, config=config).to(device)
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(resume_payload["state_dict"], strict=False)
    if args.freeze_base:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("residual_head."))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5,
    )

    for epoch in range(1, args.capacity_pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        capacity_loss = capacity_pretraining_loss(
            model, initial, controls, truth, mask,
        )
        capacity_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            print(f"pretrain={epoch:03d} capacity_loss={capacity_loss.item():.5f}")

    for epoch in range(1, args.residual_pretrain_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual_loss = residual_pretraining_loss(
            model, initial, controls, truth, mask,
        )
        residual_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            print(f"residual_pretrain={epoch:03d} residual_loss={residual_loss.item():.5f}")

    # Start long-horizon scheduling from the post-pretraining learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5,
    )

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model.rollout(initial, controls)
        loss = masked_loss(output["temperature"], truth, mask)
        # The standard SHR range 0.8-0.9 corresponds to roughly 0.050-0.056.
        # This weak prior prevents arbitrary load/capacity cancellation while
        # still allowing initialization observations to determine the value.
        load_prior = (output["load_coefficient"] - 0.053).square().mean()
        if args.capacity_aux_weight > 0:
            capacity_aux = capacity_pretraining_loss(
                model, initial, controls, truth, mask,
            )
        else:
            capacity_aux = loss.new_zeros(())
        total_loss = (
            loss + 2.0 * load_prior
            + args.capacity_aux_weight * capacity_aux
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step(float(loss.detach()))
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().cpu().clone()
                          for key, tensor in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch:03d} open_loop_loss={value:.5f} "
                  f"capacity_aux={capacity_aux.item():.5f} "
                  f"load_coeff={output['load_coefficient'].mean().item():.5f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    if best_state is None:
        raise RuntimeError("training did not produce a model")
    model.load_state_dict(best_state)
    metadata = {
        "version": "v4_hybrid_virtual_load",
        "seed": args.seed,
        "capacity_pretrain_epochs": args.capacity_pretrain_epochs,
        "resume_from": None if args.resume is None else str(args.resume),
        "training_runs": len(training_runs),
        "all_runs": len(runs),
        "best_training_open_loop_loss": best_loss,
        "data_dir": str(args.data_dir),
        "target_excluded": True,
        "condition_id_excluded": True,
        "outdoor_temperature_fixed_after_reset": True,
    }
    save_payload(args.output, model, normalization, metadata)
    manifest = describe_runs(runs)
    manifest_path = args.output.with_suffix(".runs.csv")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
