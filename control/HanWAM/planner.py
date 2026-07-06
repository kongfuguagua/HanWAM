"""Sampling-based MPC planner for HanWAM latent rollouts."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import torch

from .type import WAM_ACTION_COLS


@dataclass
class PlannerResult:
    action: np.ndarray
    best_sequence: np.ndarray
    debug: dict
    history: pd.DataFrame


class CEMPlanner:
    def __init__(
        self,
        model,
        obs_norm,
        action_norm,
        physical_norm,
        obs_cols: list[str],
        physical_cols: list[str],
        action_bounds: dict[str, tuple[float, float]],
        config: dict,
        device: torch.device,
    ):
        self.model = model
        self.obs_norm = obs_norm
        self.action_norm = action_norm
        self.physical_norm = physical_norm
        self.obs_cols = obs_cols
        self.physical_cols = physical_cols
        self.action_bounds = action_bounds
        self.config = config
        self.device = device
        self.horizon_steps = int(config.get("horizon_steps", 120))
        self.chunk_steps = int(config.get("chunk_steps", 12))
        self.num_samples = int(config.get("num_samples", 512))
        self.num_iterations = int(config.get("num_iterations", 4))
        self.elite_ratio = float(config.get("elite_ratio", 0.1))
        self.cost_weights = dict(config.get("cost_weights") or {})
        self.temperature = float(config.get("temperature", 1.0))
        self.step_seconds = int(config.get("step_seconds", 5))
        self.comfort_band_c = float(config.get("comfort_band_c", 0.5))
        self.objective = str(config.get("objective", "hanwam_simple"))
        self.reference_schedule = str(config.get("reference_schedule", "deadline_linear"))
        self.compressor_on_threshold_hz = float(config.get("compressor_on_threshold_hz", 15.0))
        self.snap_deadband_freq = bool(config.get("snap_deadband_freq", self.objective == "hanwam_simple"))
        terminal_band = config.get("terminal_comfort_band_c")
        self.terminal_comfort_band_c = None if terminal_band is None else float(terminal_band)
        terminal_margin_seconds = config.get("terminal_margin_seconds")
        self.terminal_margin_seconds = None if terminal_margin_seconds is None else float(terminal_margin_seconds)
        self.deadline_fraction = float(config.get("deadline_fraction", 0.3))
        self.temperature_source = str(config.get("temperature_source", "delta"))
        self.seed = config.get("seed")
        self._action_cols = list(WAM_ACTION_COLS)
        self._lo = np.asarray([self.action_bounds[col][0] for col in self._action_cols], dtype=np.float32)
        self._hi = np.asarray([self.action_bounds[col][1] for col in self._action_cols], dtype=np.float32)
        self._range = np.maximum(self._hi - self._lo, 1e-6)
        self._action_mean = torch.as_tensor(self.action_norm.mean, dtype=torch.float32, device=device)
        self._action_std = torch.as_tensor(self.action_norm.std, dtype=torch.float32, device=device)
        self._phys_mean = torch.as_tensor(self.physical_norm.mean, dtype=torch.float32, device=device)
        self._phys_std = torch.as_tensor(self.physical_norm.std, dtype=torch.float32, device=device)
        self._generator = None
        self.reset()

    @property
    def num_chunks(self) -> int:
        return int(np.ceil(self.horizon_steps / self.chunk_steps))

    def reset(self) -> None:
        if self.seed is None:
            self._generator = None
            return
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(self.seed))

    def _initial_distribution(self, observation: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        current = []
        obs_index = {col: idx for idx, col in enumerate(self.obs_cols)}
        for col, lo, hi in zip(self._action_cols, self._lo, self._hi):
            value = observation[obs_index[col]] if col in obs_index else (lo + hi) * 0.5
            current.append(float(np.clip(value, lo, hi)))
        mean = np.tile(np.asarray(current, dtype=np.float32), (self.num_chunks, 1))
        std = np.tile(self._range * 0.35, (self.num_chunks, 1))
        return (
            torch.as_tensor(mean, dtype=torch.float32, device=self.device),
            torch.as_tensor(std, dtype=torch.float32, device=self.device),
        )

    def _expand_chunks(self, chunk_actions: torch.Tensor) -> torch.Tensor:
        actions = chunk_actions.repeat_interleave(self.chunk_steps, dim=1)
        return actions[:, : self.horizon_steps, :]

    def _project_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Project sampled actions into the physical actuator domain."""
        if not self.snap_deadband_freq:
            return actions
        freq_idx = self._action_cols.index("freq_target")
        projected = actions.clone()
        freq = projected[..., freq_idx]
        threshold = torch.as_tensor(
            self.compressor_on_threshold_hz,
            dtype=actions.dtype,
            device=actions.device,
        )
        projected[..., freq_idx] = torch.where(
            (freq > 0.0) & (freq < threshold),
            torch.zeros_like(freq),
            freq,
        )
        return projected

    def _decode_physical(self, physical_n: torch.Tensor) -> torch.Tensor:
        return physical_n * self._phys_std.view(1, 1, -1) + self._phys_mean.view(1, 1, -1)

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self._action_mean.view(1, 1, -1)) / self._action_std.view(1, 1, -1)

    def _cost(
        self,
        physical: torch.Tensor,
        actions: torch.Tensor,
        target: float,
        initial_t_in: float,
        remaining_seconds: float | None,
        observation: np.ndarray,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        idx = {name: i for i, name in enumerate(self.physical_cols)}
        if self.temperature_source == "delta" and "T_in_delta" in idx:
            delta = physical[:, :, idx["T_in_delta"]]
            temp = torch.as_tensor(float(initial_t_in), dtype=delta.dtype, device=delta.device) + torch.cumsum(delta, dim=1)
        else:
            temp = physical[:, :, idx["T_in"]]
        energy = torch.clamp(physical[:, :, idx["electric_kwh_delta"]], min=0.0)
        target_t = torch.as_tensor(float(target), dtype=temp.dtype, device=temp.device)
        if self.reference_schedule == "deadline_linear" and remaining_seconds is not None:
            effective_remaining = max(
                float(remaining_seconds) * self.deadline_fraction,
                float(self.horizon_steps * self.step_seconds),
            )
            times = torch.arange(1, self.horizon_steps + 1, dtype=temp.dtype, device=temp.device) * float(self.step_seconds)
            progress = torch.clamp(times / max(effective_remaining, 1.0), min=0.0, max=1.0)
            ref = torch.as_tensor(float(initial_t_in), dtype=temp.dtype, device=temp.device) + (
                target_t - float(initial_t_in)
            ) * progress
        else:
            ref = torch.full((self.horizon_steps,), float(target), dtype=temp.dtype, device=temp.device)
        tracking = (temp - ref.view(1, -1)).pow(2).mean(dim=1)
        energy_sum = energy.sum(dim=1)
        action_range = torch.as_tensor(self._range, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
        if actions.shape[1] > 1:
            seq_smooth = ((actions[:, 1:] - actions[:, :-1]) / action_range).pow(2).mean(dim=(1, 2))
        else:
            seq_smooth = torch.zeros_like(tracking)
        obs_index = {col: idx for idx, col in enumerate(self.obs_cols)}
        current_action = torch.as_tensor(
            [float(observation[obs_index[col]]) for col in self._action_cols],
            dtype=actions.dtype,
            device=actions.device,
        ).view(1, 1, -1)
        first_smooth = ((actions[:, :1, :] - current_action) / action_range).pow(2).mean(dim=(1, 2))
        action_smooth = 0.5 * seq_smooth + 0.5 * first_smooth
        w = self.cost_weights
        parts = {
            "tracking": tracking * float(w.get("tracking", 4.0)),
            "energy": energy_sum * float(w.get("energy", 2.0)),
            "action_smooth": action_smooth * float(w.get("action_smooth", 0.05)),
        }
        fan_weight = float(w.get("fan", 0.0))
        off_fan_weight = float(w.get("off_fan", 0.0))
        action_prior_weight = float(w.get("action_prior", 0.0))
        if fan_weight > 0.0 or off_fan_weight > 0.0:
            freq_idx = self._action_cols.index("freq_target")
            fan_idx = self._action_cols.index("fan_out")
            fan_level = ((actions[:, :, fan_idx] - self._lo[fan_idx]) / max(float(self._range[fan_idx]), 1e-6)).clamp(0.0, 1.0)
            if fan_weight > 0.0:
                parts["fan"] = fan_level.pow(2).mean(dim=1) * fan_weight
            if off_fan_weight > 0.0:
                compressor_off = actions[:, :, freq_idx] < float(self.compressor_on_threshold_hz)
                off_fan = fan_level.pow(2) * compressor_off.to(dtype=fan_level.dtype)
                parts["off_fan"] = off_fan.mean(dim=1) * off_fan_weight
        if action_prior_weight > 0.0:
            action_z = self._normalize_actions(actions)
            parts["action_prior"] = action_z.pow(2).mean(dim=(1, 2)) * action_prior_weight
        return sum(parts.values()), parts

    @torch.no_grad()
    def plan(
        self,
        observation: np.ndarray,
        target: float,
        remaining_seconds: float | None = None,
        obs_history: np.ndarray | None = None,
    ) -> PlannerResult:
        start = time.perf_counter()
        if obs_history is not None:
            obs_input = np.asarray(obs_history, dtype=np.float32)
            obs_n = torch.as_tensor(self.obs_norm.encode(obs_input)[None], dtype=torch.float32, device=self.device)
            latent = self.model.encode(obs_n)
        else:
            obs_n = torch.as_tensor(self.obs_norm.encode(observation[None]), dtype=torch.float32, device=self.device)
            latent = self.model.encode(obs_n)
        mean, std = self._initial_distribution(observation)
        lo = torch.as_tensor(self._lo, dtype=torch.float32, device=self.device).view(1, 1, -1)
        hi = torch.as_tensor(self._hi, dtype=torch.float32, device=self.device).view(1, 1, -1)
        elite_n = max(1, int(round(self.num_samples * self.elite_ratio)))
        history_rows = []
        best_cost = None
        best_seq = None
        best_parts = None
        initial_t_in = float(observation[self.obs_cols.index("T_in")])

        for iteration in range(self.num_iterations):
            noise = torch.randn(
                self.num_samples,
                self.num_chunks,
                len(self._action_cols),
                dtype=torch.float32,
                device=self.device,
                generator=self._generator,
            )
            chunks = torch.clamp(mean.unsqueeze(0) + noise * std.unsqueeze(0), min=lo, max=hi)
            chunks = self._project_actions(chunks)
            actions = self._expand_chunks(chunks)
            actions_n = self._normalize_actions(actions)
            latents = latent.expand(self.num_samples, -1)
            _, physical_n = self.model.rollout_from_latent(latents, actions_n)
            physical = self._decode_physical(physical_n)
            cost, parts = self._cost(physical, actions, target, initial_t_in, remaining_seconds, observation)
            elite_cost, elite_idx = torch.topk(cost, elite_n, largest=False)
            elite = chunks[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, unbiased=False).clamp_min(torch.as_tensor(self._range, device=self.device).view(1, -1) * 0.03)
            current_best = int(torch.argmin(cost).item())
            if best_cost is None or float(cost[current_best]) < best_cost:
                best_cost = float(cost[current_best].cpu())
                best_seq = actions[current_best].detach().cpu().numpy()
                best_parts = {name: float(value[current_best].detach().cpu()) for name, value in parts.items()}
            history_rows.append(
                {
                    "iteration": iteration,
                    "best_cost": float(elite_cost[0].detach().cpu()),
                    "mean_elite_cost": float(elite_cost.mean().detach().cpu()),
                }
            )

        assert best_seq is not None and best_parts is not None
        plan_ms = (time.perf_counter() - start) * 1000.0
        action = np.clip(best_seq[0], self._lo, self._hi).astype(np.float32)
        debug = {
            "controller": "hanwam",
            "hanwam_planner": "cem",
            "hanwam_best_cost": float(best_cost),
            "hanwam_plan_ms": float(plan_ms),
            "hanwam_raw_freq_target": float(best_seq[0, 0]),
            "hanwam_raw_eev": float(best_seq[0, 1]),
            "hanwam_raw_fan_out": float(best_seq[0, 2]),
            "hanwam_clipped_freq_target": float(action[0]),
            "hanwam_clipped_eev": float(action[1]),
            "hanwam_clipped_fan_out": float(action[2]),
            "hanwam_planner_seed": np.nan if self.seed is None else int(self.seed),
        }
        for name, value in best_parts.items():
            debug[f"hanwam_cost_{name}"] = value
        return PlannerResult(action=action, best_sequence=best_seq, debug=debug, history=pd.DataFrame(history_rows))
