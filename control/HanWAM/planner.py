"""Sampling-based MPC planners for HanWAM latent rollouts."""
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


@dataclass(frozen=True)
class _FirstPrinciplesObjectiveConfig:
    action_history_ema_tau_seconds: float | None
    slew_pre_deadline: tuple[float, float, float] | None
    slew_post_deadline: tuple[float, float, float] | None

    @staticmethod
    def _section(payload: dict, name: str) -> dict:
        value = payload.get(name)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _optional_float(value) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _optional_slew(value) -> tuple[float, float, float] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            freq = value.get("freq", value.get("freq_target"))
            eev = value.get("eev")
            fan = value.get("fan_out", value.get("fan"))
            if freq is None or eev is None or fan is None:
                return None
            raw = [freq, eev, fan]
        else:
            raw = list(value)
            if len(raw) != 3:
                return None
        parsed = tuple(float(v) for v in raw)
        if any(v <= 0.0 for v in parsed):
            return None
        return parsed

    @classmethod
    def from_planner_config(cls, config: dict) -> "_FirstPrinciplesObjectiveConfig":
        fp = cls._section(config, "first_principles")
        action = cls._section(fp, "action")
        slew = cls._section(fp, "slew")
        ema_tau_seconds = fp.get(
            "action_history_ema_tau_seconds",
            action.get("history_ema_tau_seconds", config.get("action_history_ema_tau_seconds")),
        )
        pre_slew = cls._optional_slew(
            slew.get("pre_deadline", config.get("slew_pre_deadline"))
            if isinstance(slew, dict)
            else config.get("slew_pre_deadline")
        )
        post_slew = cls._optional_slew(
            slew.get("post_deadline", config.get("slew_post_deadline"))
            if isinstance(slew, dict)
            else config.get("slew_post_deadline")
        )
        return cls(
            action_history_ema_tau_seconds=cls._optional_float(ema_tau_seconds),
            slew_pre_deadline=pre_slew,
            slew_post_deadline=post_slew,
        )


class _SamplingPlannerBase:
    planner_name = "sampling"

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
        self.obs_cols = list(obs_cols)
        self.physical_cols = list(physical_cols)
        self.action_bounds = action_bounds
        self.config = dict(config)
        self.device = device
        timing = dict(config.get("timing") or {})
        sampling = dict(config.get("sampling") or {})
        actuator = dict(config.get("actuator") or {})
        self.frames_per_block = int(timing.get("frames_per_block", config.get("frames_per_block", getattr(model, "frames_per_block", 12))))
        self.future_blocks = int(timing.get("future_blocks", config.get("future_blocks", getattr(model, "future_blocks", 5))))
        self.history_blocks = int(timing.get("history_blocks", config.get("history_blocks", getattr(model, "history_blocks", 3))))
        self.horizon_steps = int(timing.get("horizon_steps", config.get("horizon_steps", self.frames_per_block * self.future_blocks)))
        self.rollout_steps = int(self.frames_per_block * self.future_blocks)
        self.physical_is_block = bool(getattr(model, "physical_is_block", False))
        if self.horizon_steps > self.rollout_steps:
            raise ValueError(
                f"planner.horizon_steps={self.horizon_steps} exceeds WM rollout "
                f"{self.rollout_steps} ({self.future_blocks} blocks x {self.frames_per_block} frames)"
            )
        self.chunk_steps = int(timing.get("chunk_steps", config.get("chunk_steps", self.frames_per_block)))
        self.control_interval_steps = int(timing.get("control_interval_steps", config.get("control_interval_steps", 1)))
        self.num_samples = int(sampling.get("num_samples", config.get("num_samples", 512)))
        self.num_iterations = int(sampling.get("num_iterations", config.get("num_iterations", 3)))
        self.cost_weights = dict(config.get("cost_weights") or {})
        self.temperature = float(sampling.get("temperature", config.get("temperature", 1.0)))
        self.sample_std_fraction = float(sampling.get("sample_std_fraction", config.get("sample_std_fraction", 0.25)))
        self.min_std_fraction = float(sampling.get("min_std_fraction", config.get("min_std_fraction", 0.02)))
        self.proposal_action_anchors = list(sampling.get("proposal_action_anchors", config.get("proposal_action_anchors") or []))
        self.step_seconds = int(timing.get("step_seconds", config.get("step_seconds", 5)))
        self.trajectory_step_seconds = float(
            self.step_seconds * self.frames_per_block if self.physical_is_block else self.step_seconds
        )
        self.objective = str(config.get("objective", "phase_energy_clamp"))
        self.compressor_on_threshold_hz = float(actuator.get("compressor_on_threshold_hz", config.get("compressor_on_threshold_hz", 15.0)))
        self.snap_deadband_freq = bool(actuator.get("snap_deadband_freq", config.get("snap_deadband_freq", True)))
        self.first_principles_config = _FirstPrinciplesObjectiveConfig.from_planner_config(config)
        self.temperature_source = str(config.get("temperature_source", "delta"))
        self.seed = sampling.get("seed", config.get("seed"))
        self._action_cols = list(WAM_ACTION_COLS)
        self._lo = np.asarray([self.action_bounds[col][0] for col in self._action_cols], dtype=np.float32)
        self._hi = np.asarray([self.action_bounds[col][1] for col in self._action_cols], dtype=np.float32)
        self._range = np.maximum(self._hi - self._lo, 1e-6)
        self._obs_index = {col: idx for idx, col in enumerate(self.obs_cols)}
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
        else:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(self.seed))

    def _current_action_from_observation(
        self,
        observation: np.ndarray,
        current_action: np.ndarray | None = None,
    ) -> np.ndarray:
        if current_action is not None:
            return np.clip(np.asarray(current_action, dtype=np.float32), self._lo, self._hi)
        values = []
        for col, lo, hi in zip(self._action_cols, self._lo, self._hi):
            if col in self._obs_index:
                value = float(observation[self._obs_index[col]])
            elif col == "freq_target" and "freq" in self._obs_index:
                value = float(observation[self._obs_index["freq"]])
            else:
                value = float((lo + hi) * 0.5)
            values.append(float(np.clip(value, lo, hi)))
        return np.asarray(values, dtype=np.float32)

    def _default_history_blocks(
        self,
        observation: np.ndarray,
        current_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        history_steps = self.history_blocks * self.frames_per_block
        obs = np.repeat(np.asarray(observation, dtype=np.float32)[None, :], history_steps, axis=0)
        act = np.repeat(np.asarray(current_action, dtype=np.float32)[None, :], history_steps, axis=0)
        return (
            obs.reshape(self.history_blocks, self.frames_per_block, len(self.obs_cols)),
            act.reshape(self.history_blocks, self.frames_per_block, len(self._action_cols)),
        )

    def _history_blocks(
        self,
        observation: np.ndarray,
        current_action: np.ndarray,
        obs_history_blocks: np.ndarray | None,
        act_history_blocks: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if obs_history_blocks is None or act_history_blocks is None:
            return self._default_history_blocks(observation, current_action)
        obs_blocks = np.asarray(obs_history_blocks, dtype=np.float32)
        act_blocks = np.asarray(act_history_blocks, dtype=np.float32)
        expected_obs = (self.history_blocks, self.frames_per_block, len(self.obs_cols))
        expected_act = (self.history_blocks, self.frames_per_block, len(self._action_cols))
        if tuple(obs_blocks.shape) != expected_obs:
            raise ValueError(f"obs_history_blocks must have shape {expected_obs}, got {tuple(obs_blocks.shape)}")
        if tuple(act_blocks.shape) != expected_act:
            raise ValueError(f"act_history_blocks must have shape {expected_act}, got {tuple(act_blocks.shape)}")
        return obs_blocks, act_blocks

    def _actions_to_chunks(self, actions: np.ndarray) -> torch.Tensor:
        actions = np.asarray(actions, dtype=np.float32)
        idx = np.minimum(np.arange(self.num_chunks) * self.chunk_steps, len(actions) - 1)
        chunks = actions[idx]
        return torch.as_tensor(chunks, dtype=torch.float32, device=self.device)

    def _initial_actions(
        self,
        observation: np.ndarray,
        current_action: np.ndarray,
    ) -> np.ndarray:
        return np.repeat(current_action.reshape(1, -1), self.horizon_steps, axis=0).astype(np.float32)

    def _proposal_anchor_chunks(self) -> torch.Tensor | None:
        if not self.proposal_action_anchors:
            return None
        rows = []
        for raw in self.proposal_action_anchors:
            if isinstance(raw, dict):
                action = [float(raw[col]) for col in self._action_cols]
            else:
                action = [float(value) for value in raw]
            if len(action) != len(self._action_cols):
                raise ValueError("Each proposal_action_anchors entry must have 3 values: freq_target, eev, fan_out")
            clipped = np.clip(np.asarray(action, dtype=np.float32), self._lo, self._hi)
            rows.append(np.tile(clipped.reshape(1, -1), (self.num_chunks, 1)))
        anchors = torch.as_tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32, device=self.device)
        return self._project_actions(anchors)

    def _expand_chunks(self, chunk_actions: torch.Tensor) -> torch.Tensor:
        if chunk_actions.ndim == 2:
            chunk_actions = chunk_actions.unsqueeze(0)
        actions = chunk_actions.repeat_interleave(self.chunk_steps, dim=1)
        return actions[:, : self.horizon_steps, :]

    def _project_actions(self, actions: torch.Tensor) -> torch.Tensor:
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

    def _sanitize_deadband_freq(self, chunks: torch.Tensor, previous: torch.Tensor | None = None) -> torch.Tensor:
        if not self.snap_deadband_freq:
            return chunks
        freq_idx = self._action_cols.index("freq_target")
        threshold = torch.as_tensor(
            self.compressor_on_threshold_hz,
            dtype=chunks.dtype,
            device=chunks.device,
        )
        projected = chunks.clone()
        freq = projected[..., freq_idx]
        if previous is None:
            projected[..., freq_idx] = torch.where(
                (freq > 0.0) & (freq < threshold),
                torch.zeros_like(freq),
                freq,
            )
            return projected
        prev_freq = previous[..., freq_idx]
        keep_on = (freq > 0.0) & (freq < threshold) & (prev_freq >= threshold)
        projected[..., freq_idx] = torch.where(
            keep_on,
            threshold.expand_as(freq),
            torch.where((freq > 0.0) & (freq < threshold), torch.zeros_like(freq), freq),
        )
        return projected

    def _slew_limits(
        self,
        remaining_seconds: float | None,
        transition_start_seconds: float,
        interval_seconds: float,
    ) -> torch.Tensor | None:
        cfg = self.first_principles_config
        if cfg.slew_pre_deadline is None and cfg.slew_post_deadline is None:
            return None
        use_post = remaining_seconds is not None and float(remaining_seconds) <= float(transition_start_seconds)
        raw = cfg.slew_post_deadline if use_post else cfg.slew_pre_deadline
        if raw is None:
            raw = cfg.slew_pre_deadline or cfg.slew_post_deadline
        if raw is None:
            return None
        scale = max(float(interval_seconds), 1e-6) / 10.0
        return torch.as_tensor(raw, dtype=torch.float32, device=self.device).view(1, -1) * float(scale)

    def _project_action_chunks(
        self,
        chunks: torch.Tensor,
        current_action: np.ndarray | None,
        remaining_seconds: float | None,
    ) -> torch.Tensor:
        squeeze = chunks.ndim == 2
        if squeeze:
            chunks = chunks.unsqueeze(0)
        lo = torch.as_tensor(self._lo, dtype=chunks.dtype, device=chunks.device).view(1, 1, -1)
        hi = torch.as_tensor(self._hi, dtype=chunks.dtype, device=chunks.device).view(1, 1, -1)
        projected = torch.clamp(chunks, min=lo, max=hi)
        projected = self._sanitize_deadband_freq(projected)
        if self.first_principles_config.slew_pre_deadline is None and self.first_principles_config.slew_post_deadline is None:
            return projected.squeeze(0) if squeeze else projected
        current = torch.as_tensor(
            self._current_action_from_observation(np.zeros(len(self.obs_cols), dtype=np.float32), current_action),
            dtype=chunks.dtype,
            device=chunks.device,
        ).view(1, -1).expand(projected.shape[0], -1)
        rows = []
        previous = current
        first_interval_seconds = max(float(self.control_interval_steps * self.step_seconds), float(self.step_seconds))
        chunk_interval_seconds = float(self.chunk_steps * self.step_seconds)
        for chunk_idx in range(projected.shape[1]):
            interval_seconds = first_interval_seconds if chunk_idx == 0 else chunk_interval_seconds
            transition_start = 0.0 if chunk_idx == 0 else float(chunk_idx * self.chunk_steps * self.step_seconds)
            limits = self._slew_limits(remaining_seconds, transition_start, interval_seconds)
            candidate = projected[:, chunk_idx, :]
            if limits is not None:
                limits = limits.to(dtype=chunks.dtype, device=chunks.device)
                candidate = torch.minimum(torch.maximum(candidate, previous - limits), previous + limits)
                freq_idx = self._action_cols.index("freq_target")
                threshold = torch.as_tensor(
                    self.compressor_on_threshold_hz,
                    dtype=chunks.dtype,
                    device=chunks.device,
                )
                wants_on = projected[:, chunk_idx, freq_idx] >= threshold
                prev_off = previous[:, freq_idx] < threshold
                crossing = wants_on & prev_off & (candidate[:, freq_idx] < threshold)
                candidate[:, freq_idx] = torch.where(crossing, threshold.expand_as(candidate[:, freq_idx]), candidate[:, freq_idx])
            candidate = self._sanitize_deadband_freq(candidate, previous=previous)
            rows.append(candidate)
            previous = candidate
        projected = torch.stack(rows, dim=1)
        return projected.squeeze(0) if squeeze else projected

    def _action_chunks(self, actions: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(self.num_chunks, dtype=torch.long, device=actions.device) * int(self.chunk_steps)
        idx = torch.clamp(idx, max=max(int(actions.shape[1]) - 1, 0))
        return actions.index_select(1, idx)

    def _recent_action_reference(
        self,
        act_history_blocks: np.ndarray | None,
        current_action: np.ndarray | None,
    ) -> np.ndarray:
        if act_history_blocks is None:
            return self._current_action_from_observation(np.zeros(len(self.obs_cols), dtype=np.float32), current_action)
        hist_array = np.asarray(act_history_blocks, dtype=np.float32).reshape(-1, len(self._action_cols))
        if hist_array.shape[0] == 0:
            return self._current_action_from_observation(np.zeros(len(self.obs_cols), dtype=np.float32), current_action)
        tail = hist_array[-max(1, min(int(self.chunk_steps), len(hist_array))) :]
        ema_tau_seconds = self.first_principles_config.action_history_ema_tau_seconds
        if ema_tau_seconds is None:
            reference = tail.mean(axis=0)
        else:
            tau = max(float(ema_tau_seconds), 1e-6)
            ages = np.arange(len(tail) - 1, -1, -1, dtype=np.float32) * float(self.step_seconds)
            weights = np.exp(-ages / tau).astype(np.float32)
            weights = weights / max(float(weights.sum()), 1e-12)
            reference = (tail * weights[:, None]).sum(axis=0)
        return np.clip(reference, self._lo, self._hi).astype(np.float32)

    def _chunk_action_cost(
        self,
        actions: torch.Tensor,
        act_history_blocks: np.ndarray | None,
        current_action: np.ndarray | None,
        first_weight: float = 1.0,
        sequence_weight: float = 1.0,
        reference: str = "history_reference",
        loss: str = "mse",
        huber_delta: float = 0.10,
    ) -> torch.Tensor:
        chunks = self._action_chunks(actions)
        action_range = torch.as_tensor(self._range, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
        if str(reference) in {"last_issued_command", "last_command", "current_action"}:
            recent = self._current_action_from_observation(np.zeros(len(self.obs_cols), dtype=np.float32), current_action)
        else:
            recent = self._recent_action_reference(act_history_blocks, current_action)
        recent_t = torch.as_tensor(recent, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
        def penalty(delta: torch.Tensor) -> torch.Tensor:
            if str(loss) != "huber":
                return delta.pow(2)
            limit = max(float(huber_delta), 1e-6)
            abs_delta = delta.abs()
            return torch.where(abs_delta <= limit, 0.5 * delta.pow(2), limit * (abs_delta - 0.5 * limit))

        first = penalty((chunks[:, :1, :] - recent_t) / action_range).mean(dim=(1, 2))
        if chunks.shape[1] > 1:
            sequence = penalty((chunks[:, 1:] - chunks[:, :-1]) / action_range).mean(dim=(1, 2))
        else:
            sequence = torch.zeros_like(first)
        return float(first_weight) * first + float(sequence_weight) * sequence

    @staticmethod
    def _batch_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 1:
            mask = mask.view(1, -1).expand_as(values)
        mask_f = mask.to(dtype=values.dtype, device=values.device)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (values * mask_f).sum(dim=1) / denom

    def _first_principles_phase_clamp_cost(
        self,
        temp: torch.Tensor,
        energy: torch.Tensor,
        actions: torch.Tensor,
        target_t: torch.Tensor,
        initial_t_in: float,
        remaining_seconds: float | None,
        act_history_blocks: np.ndarray | None,
        current_action: np.ndarray | None,
        times: torch.Tensor,
        trajectory_step_seconds: float,
        include_energy: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fp = _FirstPrinciplesObjectiveConfig._section(self.config, "first_principles")
        phase = _FirstPrinciplesObjectiveConfig._section(fp, "phase")
        energy_cfg = _FirstPrinciplesObjectiveConfig._section(fp, "energy")
        action_cfg = _FirstPrinciplesObjectiveConfig._section(fp, "action")
        weights = _FirstPrinciplesObjectiveConfig._section(fp, "weights")

        reach_band = max(float(phase.get("reach_band_c", 0.45)), 1e-6)
        reach_fraction = max(float(phase.get("reach_deadline_fraction", 0.65)), 1e-3)
        hold_activation_c = float(phase.get("hold_activation_c", 0.50))
        clamp_upper = max(float(phase.get("clamp_upper_c", 0.20)), 1e-6)
        clamp_lower = max(float(phase.get("clamp_lower_c", 0.25)), 1e-6)
        clamp_center_weight = max(float(phase.get("clamp_center_weight", 0.0)), 0.0)
        clamp_center_scale = max(float(phase.get("clamp_center_scale_c", 0.5)), 1e-6)

        if remaining_seconds is None or float(remaining_seconds) <= 0.0:
            pre_deadline = torch.zeros_like(times, dtype=torch.bool)
            post_deadline = torch.ones_like(times, dtype=torch.bool)
            reach_upper = torch.full_like(times, float(target_t) + reach_band)
        else:
            remaining = max(float(remaining_seconds), 1.0)
            pre_deadline = times < remaining
            post_deadline = ~pre_deadline
            effective_deadline = max(remaining * reach_fraction, float(trajectory_step_seconds))
            progress = torch.clamp(times / effective_deadline, min=0.0, max=1.0)
            ref = torch.as_tensor(float(initial_t_in), dtype=times.dtype, device=times.device) + (
                target_t - float(initial_t_in)
            ) * progress
            reach_upper = ref + reach_band

        current_hold = bool(float(initial_t_in) <= float(target_t) + hold_activation_c)
        predicted_reached = temp <= (target_t + hold_activation_c)
        reached_by_t = torch.cumsum(predicted_reached.to(torch.int32), dim=1) > 0
        clamp_mask = reached_by_t | post_deadline.view(1, -1)
        if current_hold:
            clamp_mask = torch.ones_like(clamp_mask, dtype=torch.bool)
        reach_mask = pre_deadline.view(1, -1) & (~clamp_mask)

        reach_err = torch.clamp(temp - reach_upper.view(1, -1), min=0.0) / reach_band
        reach = self._batch_masked_mean(reach_err.pow(2), reach_mask)

        upper_err = torch.clamp(temp - (target_t + clamp_upper), min=0.0) / clamp_upper
        lower_err = torch.clamp((target_t - clamp_lower) - temp, min=0.0) / clamp_lower
        center_err = ((temp - target_t) / clamp_center_scale).pow(2) * clamp_center_weight
        clamp = self._batch_masked_mean(upper_err.pow(2) + lower_err.pow(2) + center_err, clamp_mask)
        terminal_weight = max(float(phase.get("clamp_terminal_weight", 0.0)), 0.0)
        if terminal_weight > 0.0 and temp.shape[1] > 1:
            tail_steps = max(
                1,
                int(round(float(phase.get("clamp_terminal_tail_seconds", 120.0)) / float(trajectory_step_seconds))),
            )
            tail_steps = min(tail_steps, temp.shape[1] - 1)
            projection_steps = max(
                1.0,
                float(phase.get("clamp_terminal_projection_seconds", 600.0)) / float(trajectory_step_seconds),
            )
            tail_rate = (temp[:, -1] - temp[:, -1 - tail_steps]) / float(tail_steps)
            projected = temp[:, -1] + tail_rate * projection_steps
            terminal_upper = torch.clamp(projected - (target_t + clamp_upper), min=0.0) / clamp_upper
            terminal_lower = torch.clamp((target_t - clamp_lower) - projected, min=0.0) / clamp_lower
            terminal_active = clamp_mask.any(dim=1).to(dtype=temp.dtype, device=temp.device)
            clamp = clamp + terminal_weight * (terminal_upper.pow(2) + terminal_lower.pow(2)) * terminal_active

        post_now = remaining_seconds is None or float(remaining_seconds) <= 0.0
        if post_now:
            action_phase = "post"
        elif current_hold:
            action_phase = "hold"
        else:
            action_phase = "pre"
        first_weight = float(action_cfg.get(f"{action_phase}_first_weight", action_cfg.get("first_weight", 1.0)))
        sequence_weight = float(action_cfg.get(f"{action_phase}_sequence_weight", action_cfg.get("sequence_weight", 1.0)))
        action_weight = float(weights.get(f"action_{action_phase}", weights.get("action", 1.0)))
        action = self._chunk_action_cost(
            actions,
            act_history_blocks,
            current_action,
            first_weight=first_weight,
            sequence_weight=sequence_weight,
            reference=str(action_cfg.get("reference", "last_issued_command")),
            loss=str(action_cfg.get("loss", "huber")),
            huber_delta=float(action_cfg.get("huber_delta", 0.20)),
        )
        correction_weight = float(action_cfg.get("correction_weight", 0.0))
        if correction_weight > 0.0:
            chunks = self._action_chunks(actions)
            action_range = torch.as_tensor(self._range, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
            recent = self._current_action_from_observation(np.zeros(len(self.obs_cols), dtype=np.float32), current_action)
            recent_t = torch.as_tensor(recent, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
            first_delta = (chunks[:, :1, :] - recent_t).abs() / action_range
            if chunks.shape[1] > 1:
                seq_delta = (chunks[:, 1:, :] - chunks[:, :-1, :]).abs() / action_range
                deltas = torch.cat([first_delta, seq_delta], dim=1)
            else:
                deltas = first_delta
            if self.physical_is_block and temp.shape[1] == self.num_chunks:
                chunk_idx = torch.arange(self.num_chunks, dtype=torch.long, device=actions.device)
            else:
                chunk_idx = torch.arange(self.num_chunks, dtype=torch.long, device=actions.device) * int(self.chunk_steps)
                chunk_idx = torch.clamp(chunk_idx, max=max(int(temp.shape[1]) - 1, 0))
            chunk_temp = temp.index_select(1, chunk_idx)
            current_error = torch.full(
                (temp.shape[0], 1),
                abs(float(initial_t_in) - float(target_t)),
                dtype=temp.dtype,
                device=temp.device,
            )
            if chunks.shape[1] > 1:
                error_for_delta = torch.cat([(current_error), (chunk_temp[:, :-1] - target_t).abs()], dim=1)
            else:
                error_for_delta = current_error
            base = max(float(action_cfg.get("correction_base_norm", 0.05)), 0.0)
            gain = max(float(action_cfg.get("correction_gain_norm_per_c", 0.10)), 0.0)
            deadband = max(float(action_cfg.get("correction_deadband_c", 0.15)), 0.0)
            max_allow = max(float(action_cfg.get("correction_max_norm", 0.60)), base)
            allowance = torch.clamp(base + gain * torch.clamp(error_for_delta - deadband, min=0.0), max=max_allow)
            correction = torch.clamp(deltas - allowance.view(deltas.shape[0], deltas.shape[1], 1), min=0.0).pow(2).mean(dim=(1, 2))
            action = action + correction_weight * correction
        effort_weight = float(action_cfg.get(f"{action_phase}_effort_weight", action_cfg.get("effort_weight", 0.0)))
        if effort_weight > 0.0:
            chunks = self._action_chunks(actions)
            action_range = torch.as_tensor(self._range, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
            lo = torch.as_tensor(self._lo, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
            effort_scale = torch.as_tensor(
                [
                    float(action_cfg.get("effort_freq_weight", 1.0)),
                    float(action_cfg.get("effort_eev_weight", 0.0)),
                    float(action_cfg.get("effort_fan_weight", 0.25)),
                ],
                dtype=actions.dtype,
                device=actions.device,
            ).view(1, 1, -1)
            effort = (((chunks - lo) / action_range).clamp_min(0.0).pow(2) * effort_scale).mean(dim=(1, 2))
            action = action + effort_weight * effort
        anchor_weight = float(action_cfg.get(f"{action_phase}_anchor_weight", action_cfg.get("anchor_weight", 0.0)))
        if anchor_weight > 0.0:
            chunks = self._action_chunks(actions)
            action_range = torch.as_tensor(self._range, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
            anchor = torch.as_tensor(
                [
                    float(action_cfg.get(f"{action_phase}_anchor_freq", action_cfg.get("anchor_freq", 50.0))),
                    float(action_cfg.get(f"{action_phase}_anchor_eev", action_cfg.get("anchor_eev", 230.0))),
                    float(action_cfg.get(f"{action_phase}_anchor_fan", action_cfg.get("anchor_fan", 650.0))),
                ],
                dtype=actions.dtype,
                device=actions.device,
            ).view(1, 1, -1)
            deadband_action = torch.as_tensor(
                [
                    float(action_cfg.get("anchor_freq_deadband", 12.0)),
                    float(action_cfg.get("anchor_eev_deadband", 90.0)),
                    float(action_cfg.get("anchor_fan_deadband", 180.0)),
                ],
                dtype=actions.dtype,
                device=actions.device,
            ).view(1, 1, -1)
            anchor_scale = torch.as_tensor(
                [
                    float(action_cfg.get("anchor_freq_weight", 1.0)),
                    float(action_cfg.get("anchor_eev_weight", 0.0)),
                    float(action_cfg.get("anchor_fan_weight", 0.25)),
                ],
                dtype=actions.dtype,
                device=actions.device,
            ).view(1, 1, -1)
            excess = torch.clamp((chunks - anchor).abs() - deadband_action, min=0.0) / action_range
            gate_c = max(float(action_cfg.get("anchor_gate_c", 0.0)), 0.0)
            if gate_c > 0.0:
                if self.physical_is_block and temp.shape[1] == self.num_chunks:
                    chunk_idx = torch.arange(self.num_chunks, dtype=torch.long, device=actions.device)
                else:
                    chunk_idx = torch.arange(self.num_chunks, dtype=torch.long, device=actions.device) * int(self.chunk_steps)
                    chunk_idx = torch.clamp(chunk_idx, max=max(int(temp.shape[1]) - 1, 0))
                chunk_temp = temp.index_select(1, chunk_idx)
                current_error = torch.full(
                    (temp.shape[0], 1),
                    abs(float(initial_t_in) - float(target_t)),
                    dtype=temp.dtype,
                    device=temp.device,
                )
                if chunks.shape[1] > 1:
                    error_for_anchor = torch.cat([current_error, (chunk_temp[:, :-1] - target_t).abs()], dim=1)
                else:
                    error_for_anchor = current_error
                anchor_gate = torch.clamp(1.0 - error_for_anchor / gate_c, min=0.0, max=1.0).view(chunks.shape[0], chunks.shape[1], 1)
            else:
                anchor_gate = 1.0
            anchor = (excess.pow(2) * anchor_scale * anchor_gate).mean(dim=(1, 2))
            action = action + anchor_weight * anchor

        parts = {
            "reach": reach * float(weights.get("reach", 160.0)),
            "clamp": clamp * float(weights.get("clamp", 220.0)),
        }
        if include_energy:
            energy_ref_per_hour = float(
                fp.get(
                    "energy_reference_kwh_per_hour",
                    energy_cfg.get("reference_kwh_per_hour", self.config.get("energy_reference_kwh_per_hour", 0.6)),
                )
            )
            energy_ref = max(
                energy_ref_per_hour * float(temp.shape[1] * trajectory_step_seconds) / 3600.0,
                1e-6,
            )
            energy_weight = float(weights.get(f"energy_{action_phase}", weights.get("energy", 0.0)))
            parts["energy"] = energy.sum(dim=1) / energy_ref * energy_weight
        parts["action"] = action * action_weight
        return sum(parts.values()), parts

    def _decode_physical(self, physical_n: torch.Tensor) -> torch.Tensor:
        return physical_n * self._phys_std.view(1, 1, -1) + self._phys_mean.view(1, 1, -1)

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        shape = [1] * actions.ndim
        shape[-1] = -1
        return (actions - self._action_mean.view(*shape)) / self._action_std.view(*shape)

    def _future_blocks_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[1] < self.rollout_steps:
            pad = actions[:, -1:, :].expand(actions.shape[0], self.rollout_steps - actions.shape[1], actions.shape[-1])
            actions = torch.cat([actions, pad], dim=1)
        elif actions.shape[1] > self.rollout_steps:
            actions = actions[:, : self.rollout_steps, :]
        return actions.reshape(actions.shape[0], self.future_blocks, self.frames_per_block, actions.shape[-1])

    def _rollout_physical(
        self,
        observation: np.ndarray,
        current_action: np.ndarray,
        actions: torch.Tensor,
        obs_history_blocks: np.ndarray | None,
        act_history_blocks: np.ndarray | None,
        prepared_context=None,
    ) -> torch.Tensor:
        future_act_blocks = self._future_blocks_from_actions(actions)
        future_act_n = self._normalize_actions(future_act_blocks)
        if prepared_context is not None:
            _, physical_n = self.model.rollout_prepared(prepared_context, future_act_n)
        else:
            obs_blocks, act_blocks = self._history_blocks(
                observation,
                current_action,
                obs_history_blocks,
                act_history_blocks,
            )
            obs_n = torch.as_tensor(
                self.obs_norm.encode(obs_blocks)[None],
                dtype=torch.float32,
                device=self.device,
            ).expand(actions.shape[0], -1, -1, -1)
            act_hist_n = torch.as_tensor(
                self.action_norm.encode(act_blocks)[None],
                dtype=torch.float32,
                device=self.device,
            ).expand(actions.shape[0], -1, -1, -1)
            _, physical_n = self.model.rollout(obs_n, act_hist_n, future_act_n)
        physical = self._decode_physical(physical_n)
        limit = self.future_blocks if self.physical_is_block else self.horizon_steps
        return physical[:, :limit, :]

    def _prepare_rollout_context(
        self,
        observation: np.ndarray,
        current_action: np.ndarray,
        obs_history_blocks: np.ndarray | None,
        act_history_blocks: np.ndarray | None,
    ):
        if not hasattr(self.model, "prepare_context") or not hasattr(self.model, "rollout_prepared"):
            return None
        obs_blocks, act_blocks = self._history_blocks(
            observation,
            current_action,
            obs_history_blocks,
            act_history_blocks,
        )
        obs_n = torch.as_tensor(
            self.obs_norm.encode(obs_blocks)[None],
            dtype=torch.float32,
            device=self.device,
        )
        act_n = torch.as_tensor(
            self.action_norm.encode(act_blocks)[None],
            dtype=torch.float32,
            device=self.device,
        )
        return self.model.prepare_context(obs_n, act_n)

    def _cost(
        self,
        physical: torch.Tensor,
        actions: torch.Tensor,
        target: float,
        initial_t_in: float,
        remaining_seconds: float | None,
        observation: np.ndarray,
        current_action: np.ndarray | None = None,
        act_history_blocks: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        idx = {name: i for i, name in enumerate(self.physical_cols)}
        steps = int(physical.shape[1])
        if self.temperature_source == "delta" and "T_in_delta" in idx:
            delta = physical[:, :, idx["T_in_delta"]]
            temp = torch.as_tensor(float(initial_t_in), dtype=delta.dtype, device=delta.device) + torch.cumsum(delta, dim=1)
        else:
            temp = physical[:, :, idx["T_in"]]
        energy = torch.clamp(physical[:, :, idx["electric_kwh_delta"]], min=0.0)
        target_t = torch.as_tensor(float(target), dtype=temp.dtype, device=temp.device)
        times = torch.arange(1, steps + 1, dtype=temp.dtype, device=temp.device) * float(
            self.trajectory_step_seconds
        )
        if self.objective != "phase_energy_clamp":
            raise ValueError(
                "Unsupported HanWAM planner objective "
                f"{self.objective!r}; current code keeps only phase_energy_clamp."
            )
        return self._first_principles_phase_clamp_cost(
            temp,
            energy,
            actions,
            target_t,
            initial_t_in,
            remaining_seconds,
            act_history_blocks,
            current_action,
            times,
            trajectory_step_seconds=self.trajectory_step_seconds,
            include_energy=True,
        )

    def _evaluate_actions(
        self,
        observation: np.ndarray,
        target: float,
        remaining_seconds: float | None,
        actions: torch.Tensor,
        obs_history_blocks: np.ndarray | None,
        act_history_blocks: np.ndarray | None,
        current_action: np.ndarray,
        prepared_context=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        physical = self._rollout_physical(
            observation,
            current_action,
            actions,
            obs_history_blocks,
            act_history_blocks,
            prepared_context=prepared_context,
        )
        initial_t_in = float(observation[self._obs_index["T_in"]])
        return self._cost(
            physical,
            actions[:, : self.horizon_steps, :],
            target,
            initial_t_in,
            remaining_seconds,
            observation,
            current_action=current_action,
            act_history_blocks=act_history_blocks,
        )

    def _debug_result(
        self,
        sequence: np.ndarray,
        cost: float,
        parts: dict[str, float],
        plan_ms: float,
    ) -> tuple[np.ndarray, dict]:
        action = np.clip(sequence[0], self._lo, self._hi).astype(np.float32)
        debug = {
            "controller": "hanwam",
            "hanwam_planner": self.planner_name,
            "hanwam_best_cost": float(cost),
            "hanwam_plan_ms": float(plan_ms),
            "hanwam_raw_freq_target": float(sequence[0, 0]),
            "hanwam_raw_eev": float(sequence[0, 1]),
            "hanwam_raw_fan_out": float(sequence[0, 2]),
            "hanwam_clipped_freq_target": float(action[0]),
            "hanwam_clipped_eev": float(action[1]),
            "hanwam_clipped_fan_out": float(action[2]),
            "hanwam_planner_seed": np.nan if self.seed is None else int(self.seed),
        }
        for name, value in parts.items():
            debug[f"hanwam_cost_{name}"] = value
        return action, debug


class MPPIPlanner(_SamplingPlannerBase):
    """MPPI planner with softmax-weighted nominal action updates."""

    planner_name = "mppi"

    def reset(self) -> None:
        super().reset()
        self._nominal_actions: np.ndarray | None = None

    def _nominal_for_plan(self, observation: np.ndarray, current_action: np.ndarray) -> np.ndarray:
        if self._nominal_actions is None or len(self._nominal_actions) != self.horizon_steps:
            return self._initial_actions(observation, current_action)
        shift = int(np.clip(self.control_interval_steps, 0, self.horizon_steps - 1))
        if shift <= 0:
            shifted = self._nominal_actions.copy()
        else:
            tail = np.repeat(self._nominal_actions[-1:, :], shift, axis=0)
            shifted = np.concatenate([self._nominal_actions[shift:], tail], axis=0)
        shifted[0] = current_action
        return shifted.astype(np.float32)

    @torch.no_grad()
    def plan(
        self,
        observation: np.ndarray,
        target: float,
        remaining_seconds: float | None = None,
        obs_history_blocks: np.ndarray | None = None,
        act_history_blocks: np.ndarray | None = None,
        current_action: np.ndarray | None = None,
    ) -> PlannerResult:
        start = time.perf_counter()
        observation = np.asarray(observation, dtype=np.float32)
        current = self._current_action_from_observation(observation, current_action)
        nominal_actions = self._nominal_for_plan(observation, current)
        nominal_chunks = self._actions_to_chunks(nominal_actions)
        nominal_chunks = self._project_action_chunks(nominal_chunks, current, remaining_seconds)
        std = torch.as_tensor(
            np.tile(self._range * self.sample_std_fraction, (self.num_chunks, 1)),
            dtype=torch.float32,
            device=self.device,
        )
        min_std = torch.as_tensor(
            self._range * self.min_std_fraction,
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        lo = torch.as_tensor(self._lo, dtype=torch.float32, device=self.device).view(1, 1, -1)
        hi = torch.as_tensor(self._hi, dtype=torch.float32, device=self.device).view(1, 1, -1)
        history_rows = []
        best_cost = None
        best_seq = None
        best_parts = None
        anchor_chunks = self._proposal_anchor_chunks()
        prepared_context = self._prepare_rollout_context(
            observation,
            current,
            obs_history_blocks,
            act_history_blocks,
        )

        for iteration in range(self.num_iterations):
            noise = torch.randn(
                self.num_samples,
                self.num_chunks,
                len(self._action_cols),
                dtype=torch.float32,
                device=self.device,
                generator=self._generator,
            )
            chunks = torch.clamp(nominal_chunks.unsqueeze(0) + noise * std.unsqueeze(0), min=lo, max=hi)
            chunks[0] = nominal_chunks
            if anchor_chunks is not None and self.num_samples > 1:
                count = min(int(anchor_chunks.shape[0]), self.num_samples - 1)
                chunks[1 : 1 + count] = anchor_chunks[:count]
            chunks = self._project_action_chunks(chunks, current, remaining_seconds)
            actions = self._expand_chunks(chunks)
            cost, parts = self._evaluate_actions(
                observation,
                target,
                remaining_seconds,
                actions,
                obs_history_blocks,
                act_history_blocks,
                current,
                prepared_context=prepared_context,
            )
            scaled = -(cost - cost.min()) / max(float(self.temperature), 1e-6)
            weights = torch.softmax(scaled, dim=0)
            nominal_chunks = torch.sum(weights.view(-1, 1, 1) * chunks, dim=0)
            nominal_chunks = self._project_action_chunks(
                torch.clamp(nominal_chunks, min=lo.squeeze(0), max=hi.squeeze(0)),
                current,
                remaining_seconds,
            )
            variance = torch.sum(weights.view(-1, 1, 1) * (chunks - nominal_chunks.unsqueeze(0)).pow(2), dim=0)
            std = torch.sqrt(variance + 1e-8).clamp_min(min_std)
            current_best = int(torch.argmin(cost).item())
            current_best_cost = float(cost[current_best].detach().cpu())
            if best_cost is None or current_best_cost < best_cost:
                best_cost = current_best_cost
                best_seq = actions[current_best].detach().cpu().numpy()
                best_parts = {name: float(value[current_best].detach().cpu()) for name, value in parts.items()}
            history_rows.append(
                {
                    "iteration": iteration,
                    "best_cost": current_best_cost,
                    "mean_cost": float(cost.mean().detach().cpu()),
                    "weighted_cost": float((weights * cost).sum().detach().cpu()),
                }
            )

        nominal_actions_t = self._expand_chunks(nominal_chunks)[0:1]
        nominal_cost, nominal_parts_t = self._evaluate_actions(
            observation,
            target,
            remaining_seconds,
            nominal_actions_t,
            obs_history_blocks,
            act_history_blocks,
            current,
            prepared_context=prepared_context,
        )
        nominal_seq = nominal_actions_t[0].detach().cpu().numpy()
        self._nominal_actions = nominal_seq.astype(np.float32)
        plan_ms = (time.perf_counter() - start) * 1000.0
        parts = {name: float(value[0].detach().cpu()) for name, value in nominal_parts_t.items()}
        action, debug = self._debug_result(
            nominal_seq,
            float(nominal_cost[0].detach().cpu()),
            parts,
            plan_ms,
        )
        if best_cost is not None:
            debug["hanwam_best_sample_cost"] = float(best_cost)
        if best_parts is not None:
            for name, value in best_parts.items():
                debug[f"hanwam_best_sample_cost_{name}"] = value
        debug["hanwam_history_encode_count"] = 1 if prepared_context is not None else self.num_samples * self.num_iterations
        debug["hanwam_physical_horizon_nodes"] = self.future_blocks if self.physical_is_block else self.horizon_steps
        return PlannerResult(action=action, best_sequence=nominal_seq, debug=debug, history=pd.DataFrame(history_rows))
