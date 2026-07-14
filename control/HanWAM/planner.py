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
    band_c: float
    pre_deadline_band_c: float
    guard_seconds: float
    post_deadline_upper_c: float
    post_deadline_lower_c: float
    thermal_envelope_weight: float
    terminal_weight: float
    energy_weight: float
    action_weight: float
    action_first_weight: float
    action_sequence_weight: float
    energy_reference_kwh_per_hour: float
    terminal_tail_seconds: float
    terminal_projection_max_seconds: float
    terminal_enabled_only_before_horizon_contains_deadline: bool
    terminal_slope_method: str
    terminal_slope_clip: bool
    terminal_slope_clip_c_per_min: float
    action_reference: str
    action_loss: str
    action_huber_delta: float
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
    def from_planner_config(cls, config: dict, comfort_band_c: float) -> "_FirstPrinciplesObjectiveConfig":
        fp = cls._section(config, "first_principles")
        envelope = cls._section(fp, "envelope")
        terminal = cls._section(fp, "terminal")
        energy = cls._section(fp, "energy")
        action = cls._section(fp, "action")
        slew = cls._section(fp, "slew")
        weights = dict(config.get("cost_weights") or {})
        weights.update(cls._section(fp, "weights"))

        band_c = fp.get(
            "band_c",
            envelope.get(
                "band_c",
                weights.get("temperature_band_c", config.get("comfort_band_c", comfort_band_c)),
            ),
        )
        terminal_tail_seconds = fp.get(
            "terminal_tail_seconds",
            terminal.get("tail_seconds", config.get("terminal_guard_tail_seconds", 120.0)),
        )
        terminal_projection_max_seconds = fp.get(
            "terminal_projection_max_seconds",
            terminal.get("projection_max_seconds", config.get("terminal_projection_max_seconds", 600.0)),
        )
        energy_reference = fp.get(
            "energy_reference_kwh_per_hour",
            energy.get("reference_kwh_per_hour", config.get("energy_reference_kwh_per_hour", 0.6)),
        )
        ema_tau_seconds = fp.get(
            "action_history_ema_tau_seconds",
            action.get("history_ema_tau_seconds", config.get("action_history_ema_tau_seconds")),
        )
        pre_deadline_band = envelope.get(
            "pre_deadline_band_c",
            fp.get("pre_deadline_band_c", band_c),
        )
        post_upper = envelope.get(
            "post_deadline_upper_c",
            fp.get("post_deadline_upper_c", band_c),
        )
        post_lower = envelope.get(
            "post_deadline_lower_c",
            fp.get("post_deadline_lower_c", band_c),
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
            band_c=float(band_c),
            pre_deadline_band_c=float(pre_deadline_band),
            guard_seconds=float(envelope.get("guard_seconds", fp.get("guard_seconds", 0.0))),
            post_deadline_upper_c=float(post_upper),
            post_deadline_lower_c=float(post_lower),
            thermal_envelope_weight=float(weights.get("thermal_envelope", 0.0)),
            terminal_weight=float(weights.get("terminal", 0.0)),
            energy_weight=float(weights.get("energy", 0.0)),
            action_weight=float(weights.get("action", 0.0)),
            action_first_weight=float(weights.get("action_first_weight", action.get("first_weight", 1.0))),
            action_sequence_weight=float(weights.get("action_sequence_weight", action.get("sequence_weight", 1.0))),
            energy_reference_kwh_per_hour=float(energy_reference),
            terminal_tail_seconds=float(terminal_tail_seconds),
            terminal_projection_max_seconds=float(terminal_projection_max_seconds),
            terminal_enabled_only_before_horizon_contains_deadline=bool(
                terminal.get(
                    "enabled_only_before_horizon_contains_deadline",
                    config.get("terminal_enabled_only_before_horizon_contains_deadline", False),
                )
            ),
            terminal_slope_method=str(terminal.get("slope_method", config.get("terminal_slope_method", "endpoint"))),
            terminal_slope_clip=bool(terminal.get("slope_clip", config.get("terminal_slope_clip", False))),
            terminal_slope_clip_c_per_min=float(
                terminal.get("slope_clip_c_per_min", config.get("terminal_slope_clip_c_per_min", 0.20))
            ),
            action_reference=str(action.get("reference", config.get("action_reference", "history_reference"))),
            action_loss=str(action.get("loss", config.get("action_loss", "mse"))),
            action_huber_delta=float(action.get("huber_delta", config.get("action_huber_delta", 0.10))),
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
        self.comfort_band_c = float(config.get("comfort_band_c", 0.5))
        self.comfort_band_reference = str(config.get("comfort_band_reference", "target"))
        self.objective = str(config.get("objective", "hanwam_e007"))
        self.reference_schedule = str(config.get("reference_schedule", "deadline_linear"))
        self.compressor_on_threshold_hz = float(actuator.get("compressor_on_threshold_hz", config.get("compressor_on_threshold_hz", 15.0)))
        self.snap_deadband_freq = bool(actuator.get("snap_deadband_freq", config.get("snap_deadband_freq", True)))
        target_band_margin_seconds = config.get("target_band_margin_seconds")
        self.target_band_margin_seconds = (
            None if target_band_margin_seconds is None else float(target_band_margin_seconds)
        )
        deadline_band = config.get("deadline_comfort_band_c")
        self.deadline_comfort_band_c = None if deadline_band is None else float(deadline_band)
        target_margin_band = config.get("target_margin_comfort_band_c")
        self.target_margin_comfort_band_c = None if target_margin_band is None else float(target_margin_band)
        self.deadline_fraction = float(config.get("deadline_fraction", 0.3))
        self.terminal_guard_band_c = float(
            config.get(
                "terminal_guard_band_c",
                self.target_margin_comfort_band_c
                if self.target_margin_comfort_band_c is not None
                else self.comfort_band_c,
            )
        )
        self.terminal_guard_tail_seconds = float(config.get("terminal_guard_tail_seconds", 120.0))
        self.terminal_guard_projection_seconds = float(config.get("terminal_guard_projection_seconds", 120.0))
        self.thermal_envelope_lower_weight = float(config.get("thermal_envelope_lower_weight", 1.0))
        upper_reserve_c = config.get("upper_reserve_c")
        self.upper_reserve_c = None if upper_reserve_c is None else float(upper_reserve_c)
        self.first_principles_config = _FirstPrinciplesObjectiveConfig.from_planner_config(config, self.comfort_band_c)
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

    def _first_principles_bounds(
        self,
        times: torch.Tensor,
        initial_t_in: float,
        target_t: torch.Tensor,
        remaining_seconds: float | None,
        band_c: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if remaining_seconds is None or float(remaining_seconds) <= 0.0:
            upper = target_t + float(band_c)
            lower = target_t - float(band_c)
            return (
                upper.expand_as(times),
                lower.expand_as(times),
                torch.ones_like(times, dtype=torch.bool),
            )
        remaining = max(float(remaining_seconds), 1.0)
        progress = torch.clamp(times / remaining, min=0.0, max=1.0)
        ref = torch.as_tensor(float(initial_t_in), dtype=times.dtype, device=times.device) + (
            target_t - float(initial_t_in)
        ) * progress
        post_deadline = times >= remaining
        upper = torch.where(post_deadline, target_t + float(band_c), ref + float(band_c))
        lower = torch.where(
            post_deadline,
            target_t - float(band_c),
            torch.full_like(times, -1.0e6),
        )
        return upper, lower, post_deadline

    def _first_principles_future_upper(
        self,
        future_seconds: torch.Tensor,
        initial_t_in: float,
        target_t: torch.Tensor,
        remaining_seconds: float | None,
        band_c: float,
    ) -> torch.Tensor:
        if remaining_seconds is None or float(remaining_seconds) <= 0.0:
            return target_t + float(band_c)
        remaining = max(float(remaining_seconds), 1.0)
        progress = torch.clamp(future_seconds / remaining, min=0.0, max=1.0)
        ref = torch.as_tensor(float(initial_t_in), dtype=future_seconds.dtype, device=future_seconds.device) + (
            target_t - float(initial_t_in)
        ) * progress
        return torch.where(future_seconds >= remaining, target_t + float(band_c), ref + float(band_c))

    def _first_principles_cost(
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.first_principles_config
        band_c = float(cfg.band_c)
        band_scale = max(band_c, 1e-6)
        upper, lower, _ = self._first_principles_bounds(
            times,
            initial_t_in,
            target_t,
            remaining_seconds,
            band_c,
        )
        upper_err = torch.clamp(temp - upper.view(1, -1), min=0.0)
        lower_err = torch.clamp(lower.view(1, -1) - temp, min=0.0)
        thermal_envelope = (upper_err.pow(2) + lower_err.pow(2)).mean(dim=1) / (band_scale * band_scale)

        terminal = torch.zeros_like(thermal_envelope)
        if temp.shape[1] > 1:
            tail_steps = max(1, int(round(float(cfg.terminal_tail_seconds) / float(self.step_seconds))))
            tail_steps = min(tail_steps, temp.shape[1] - 1)
            terminal_rate_per_step = (temp[:, -1] - temp[:, -1 - tail_steps]) / float(tail_steps)
            horizon_seconds = float(temp.shape[1] * self.step_seconds)
            if remaining_seconds is None:
                projection_seconds = float(cfg.terminal_projection_max_seconds)
            else:
                remaining_after_horizon = float(remaining_seconds) - horizon_seconds
                projection_seconds = (
                    min(remaining_after_horizon, float(cfg.terminal_projection_max_seconds))
                    if remaining_after_horizon > 0.0
                    else float(cfg.terminal_projection_max_seconds)
                )
            projection_steps = max(1.0, projection_seconds / float(self.step_seconds))
            future_seconds = times[-1] + temp.new_tensor(float(projection_seconds))
            future_upper = self._first_principles_future_upper(
                future_seconds,
                initial_t_in,
                target_t,
                remaining_seconds,
                band_c,
            )
            projected_temp = temp[:, -1] + terminal_rate_per_step * projection_steps
            terminal = torch.clamp(projected_temp - future_upper, min=0.0).pow(2) / (band_scale * band_scale)

        energy_ref = max(
            float(cfg.energy_reference_kwh_per_hour) * float(temp.shape[1] * self.step_seconds) / 3600.0,
            1e-6,
        )
        energy_normalized = energy.sum(dim=1) / energy_ref

        action = self._chunk_action_cost(
            actions,
            act_history_blocks,
            current_action,
            first_weight=float(cfg.action_first_weight),
            sequence_weight=float(cfg.action_sequence_weight),
        )
        parts = {
            "thermal_envelope": thermal_envelope * float(cfg.thermal_envelope_weight),
            "terminal": terminal * float(cfg.terminal_weight),
            "energy": energy_normalized * float(cfg.energy_weight),
            "action": action * float(cfg.action_weight),
        }
        return sum(parts.values()), parts

    def _first_principles_robust_bounds(
        self,
        times: torch.Tensor,
        initial_t_in: float,
        target_t: torch.Tensor,
        remaining_seconds: float | None,
        cfg: _FirstPrinciplesObjectiveConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if remaining_seconds is None or float(remaining_seconds) <= 0.0:
            upper = target_t + float(cfg.post_deadline_upper_c)
            lower = target_t - float(cfg.post_deadline_lower_c)
            return upper.expand_as(times), lower.expand_as(times), torch.ones_like(times)
        remaining = max(float(remaining_seconds), 1.0)
        progress = torch.clamp(times / remaining, min=0.0, max=1.0)
        ref = torch.as_tensor(float(initial_t_in), dtype=times.dtype, device=times.device) + (
            target_t - float(initial_t_in)
        ) * progress
        pre_upper = ref + float(cfg.pre_deadline_band_c)
        post_upper = target_t + float(cfg.post_deadline_upper_c)
        post_lower = target_t - float(cfg.post_deadline_lower_c)
        post_deadline = times >= remaining
        guard = max(float(cfg.guard_seconds), 0.0)
        if guard > 0.0:
            guard_start = max(remaining - guard, 0.0)
            guard_width = max(remaining - guard_start, 1.0)
            guard_fraction = torch.clamp((times - guard_start) / guard_width, min=0.0, max=1.0)
        else:
            guard_fraction = torch.zeros_like(times)
        upper = torch.where(
            post_deadline,
            post_upper.expand_as(times),
            pre_upper * (1.0 - guard_fraction) + post_upper * guard_fraction,
        )
        lower_weight = torch.where(post_deadline, torch.ones_like(times), guard_fraction)
        lower = torch.where(
            (lower_weight > 0.0) | post_deadline,
            post_lower.expand_as(times),
            torch.full_like(times, -1.0e6),
        )
        return upper, lower, lower_weight

    def _first_principles_robust_future_upper(
        self,
        future_seconds: torch.Tensor,
        initial_t_in: float,
        target_t: torch.Tensor,
        remaining_seconds: float | None,
        cfg: _FirstPrinciplesObjectiveConfig,
    ) -> torch.Tensor:
        upper, _, _ = self._first_principles_robust_bounds(
            future_seconds.reshape(-1),
            initial_t_in,
            target_t,
            remaining_seconds,
            cfg,
        )
        return upper.reshape(future_seconds.shape)

    def _terminal_temperature_slope(self, temp: torch.Tensor, tail_steps: int, cfg: _FirstPrinciplesObjectiveConfig) -> torch.Tensor:
        tail_steps = min(max(1, int(tail_steps)), temp.shape[1] - 1)
        if str(cfg.terminal_slope_method) in {"robust_linear", "linear"} and tail_steps >= 2:
            tail = temp[:, -tail_steps:]
            x = torch.arange(tail.shape[1], dtype=temp.dtype, device=temp.device)
            x = x - x.mean()
            denom = x.pow(2).sum().clamp_min(1e-6)
            y = tail - tail.mean(dim=1, keepdim=True)
            slope = (y * x.view(1, -1)).sum(dim=1) / denom
        else:
            slope = (temp[:, -1] - temp[:, -1 - tail_steps]) / float(tail_steps)
        if bool(cfg.terminal_slope_clip):
            max_per_step = float(cfg.terminal_slope_clip_c_per_min) * float(self.step_seconds) / 60.0
            slope = torch.clamp(slope, min=-max_per_step, max=max_per_step)
        return slope

    def _first_principles_robust_cost(
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.first_principles_config
        band_scale = max(float(cfg.pre_deadline_band_c), float(cfg.post_deadline_upper_c), float(cfg.post_deadline_lower_c), 1e-6)
        upper, lower, lower_weight = self._first_principles_robust_bounds(
            times,
            initial_t_in,
            target_t,
            remaining_seconds,
            cfg,
        )
        upper_err = torch.clamp(temp - upper.view(1, -1), min=0.0)
        lower_err = torch.clamp(lower.view(1, -1) - temp, min=0.0) * lower_weight.view(1, -1)
        thermal_envelope = (upper_err.pow(2) + lower_err.pow(2)).mean(dim=1) / (band_scale * band_scale)

        terminal = torch.zeros_like(thermal_envelope)
        horizon_seconds = float(temp.shape[1] * self.step_seconds)
        terminal_allowed = remaining_seconds is not None and float(remaining_seconds) > horizon_seconds
        if (
            temp.shape[1] > 1
            and terminal_allowed
            and (not cfg.terminal_enabled_only_before_horizon_contains_deadline or float(remaining_seconds) > horizon_seconds)
        ):
            projection_seconds = min(
                float(remaining_seconds) - horizon_seconds,
                float(cfg.terminal_projection_max_seconds),
            )
            if projection_seconds > 0.0:
                tail_steps = max(1, int(round(float(cfg.terminal_tail_seconds) / float(self.step_seconds))))
                tail_steps = min(tail_steps, temp.shape[1] - 1)
                terminal_rate_per_step = self._terminal_temperature_slope(temp, tail_steps, cfg)
                projection_steps = max(1.0, projection_seconds / float(self.step_seconds))
                future_seconds = times[-1] + temp.new_tensor(float(projection_seconds))
                future_upper = self._first_principles_robust_future_upper(
                    future_seconds,
                    initial_t_in,
                    target_t,
                    remaining_seconds,
                    cfg,
                )
                projected_temp = temp[:, -1] + terminal_rate_per_step * projection_steps
                terminal = torch.clamp(projected_temp - future_upper, min=0.0).pow(2) / (band_scale * band_scale)

        energy_ref = max(
            float(cfg.energy_reference_kwh_per_hour) * float(temp.shape[1] * self.step_seconds) / 3600.0,
            1e-6,
        )
        energy_normalized = energy.sum(dim=1) / energy_ref
        action = self._chunk_action_cost(
            actions,
            act_history_blocks,
            current_action,
            first_weight=float(cfg.action_first_weight),
            sequence_weight=float(cfg.action_sequence_weight),
            reference=str(cfg.action_reference),
            loss=str(cfg.action_loss),
            huber_delta=float(cfg.action_huber_delta),
        )
        parts = {
            "thermal_envelope": thermal_envelope * float(cfg.thermal_envelope_weight),
            "terminal": terminal * float(cfg.terminal_weight),
            "energy": energy_normalized * float(cfg.energy_weight),
            "action": action * float(cfg.action_weight),
        }
        return sum(parts.values()), parts

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
            effective_deadline = max(remaining * reach_fraction, float(self.step_seconds))
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
            tail_steps = max(1, int(round(float(phase.get("clamp_terminal_tail_seconds", 120.0)) / float(self.step_seconds))))
            tail_steps = min(tail_steps, temp.shape[1] - 1)
            projection_steps = max(
                1.0,
                float(phase.get("clamp_terminal_projection_seconds", 600.0)) / float(self.step_seconds),
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
                energy_ref_per_hour * float(temp.shape[1] * self.step_seconds) / 3600.0,
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
    ) -> torch.Tensor:
        obs_blocks, act_blocks = self._history_blocks(observation, current_action, obs_history_blocks, act_history_blocks)
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
        future_act_blocks = self._future_blocks_from_actions(actions)
        future_act_n = self._normalize_actions(future_act_blocks)
        _, physical_n = self.model.rollout(obs_n, act_hist_n, future_act_n)
        physical = self._decode_physical(physical_n)
        return physical[:, : self.horizon_steps, :]

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
        steps = int(actions.shape[1])
        if self.temperature_source == "delta" and "T_in_delta" in idx:
            delta = physical[:, :, idx["T_in_delta"]]
            temp = torch.as_tensor(float(initial_t_in), dtype=delta.dtype, device=delta.device) + torch.cumsum(delta, dim=1)
        else:
            temp = physical[:, :, idx["T_in"]]
        energy = torch.clamp(physical[:, :, idx["electric_kwh_delta"]], min=0.0)
        target_t = torch.as_tensor(float(target), dtype=temp.dtype, device=temp.device)
        times = torch.arange(1, steps + 1, dtype=temp.dtype, device=temp.device) * float(self.step_seconds)
        if self.objective in {"hanwam_e060_phase_energy_clamp_v1", "phase_energy_clamp"}:
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
                include_energy=True,
            )
        if self.objective in {"hanwam_e057_phase_aware_clamp_v1", "phase_aware_clamp"}:
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
            )
        if self.objective in {"hanwam_e056_first_principles_robust_envelope_v1", "first_principles_robust_envelope"}:
            return self._first_principles_robust_cost(
                temp,
                energy,
                actions,
                target_t,
                initial_t_in,
                remaining_seconds,
                act_history_blocks,
                current_action,
                times,
            )
        if self.objective in {"hanwam_e055_first_principles_envelope_v1", "first_principles_envelope"}:
            return self._first_principles_cost(
                temp,
                energy,
                actions,
                target_t,
                initial_t_in,
                remaining_seconds,
                act_history_blocks,
                current_action,
                times,
            )
        if self.reference_schedule == "deadline_linear" and remaining_seconds is not None:
            effective_remaining = max(
                float(remaining_seconds) * self.deadline_fraction,
                float(steps * self.step_seconds),
            )
            progress = torch.clamp(times / max(effective_remaining, 1.0), min=0.0, max=1.0)
            ref = torch.as_tensor(float(initial_t_in), dtype=temp.dtype, device=temp.device) + (
                target_t - float(initial_t_in)
            ) * progress
        else:
            ref = torch.full((steps,), float(target), dtype=temp.dtype, device=temp.device)
        if self.comfort_band_reference in {"reference", "ref", "schedule", "deadline_linear"}:
            comfort_ref = ref.view(1, -1)
        elif self.comfort_band_reference in {"schedule_then_target", "deadline_then_target"}:
            comfort_ref = ref.view(1, -1).expand_as(temp)
            if remaining_seconds is not None:
                margin = float(self.target_band_margin_seconds or 0.0)
                target_zone = (float(remaining_seconds) - times) <= margin
                comfort_ref = torch.where(target_zone.view(1, -1), target_t.expand_as(temp), comfort_ref)
        else:
            comfort_ref = target_t
        comfort_band_violation = torch.clamp((temp - comfort_ref).abs() - float(self.comfort_band_c), min=0.0).pow(2).mean(dim=1)
        target_abs_error = (temp - target_t).abs()
        deadline_band_c = float(
            self.deadline_comfort_band_c if self.deadline_comfort_band_c is not None else self.comfort_band_c
        )
        target_margin_band_c = float(
            self.target_margin_comfort_band_c
            if self.target_margin_comfort_band_c is not None
            else self.comfort_band_c
        )
        deadline_band_error = torch.clamp(target_abs_error - deadline_band_c, min=0.0).pow(2)
        target_margin_error = torch.clamp(target_abs_error - target_margin_band_c, min=0.0).pow(2)
        if remaining_seconds is not None:
            remaining = float(remaining_seconds)
            post_deadline_mask = (remaining - times) <= 0.0
            margin = float(self.target_band_margin_seconds or 0.0)
            target_margin_mask = (remaining - times) <= margin
        else:
            post_deadline_mask = torch.zeros_like(times, dtype=torch.bool)
            target_margin_mask = torch.zeros_like(times, dtype=torch.bool)

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask_f = mask.to(dtype=values.dtype, device=values.device).view(1, -1)
            denom = mask_f.sum().clamp_min(1.0)
            return (values * mask_f).sum(dim=1) / denom

        deadline_band_violation = masked_mean(deadline_band_error, post_deadline_mask)
        target_margin_band_violation = masked_mean(target_margin_error, target_margin_mask)
        target_center = masked_mean((temp - target_t).pow(2), target_margin_mask)
        if temp.shape[1] > 1:
            temp_rate_error = (temp[:, 1:] - temp[:, :-1]).pow(2)
            temp_rate_mask = target_margin_mask[1:] | target_margin_mask[:-1]
            temperature_rate = masked_mean(temp_rate_error, temp_rate_mask)
        else:
            temperature_rate = torch.zeros_like(deadline_band_violation)
        energy_sum = energy.sum(dim=1)
        action_range = torch.as_tensor(self._range, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
        if actions.shape[1] > 1:
            seq_smooth = ((actions[:, 1:] - actions[:, :-1]) / action_range).pow(2).mean(dim=(1, 2))
        else:
            seq_smooth = torch.zeros_like(energy_sum)
        current = self._current_action_from_observation(observation, current_action)
        current_action_t = torch.as_tensor(current, dtype=actions.dtype, device=actions.device).view(1, 1, -1)
        first_smooth = ((actions[:, :1, :] - current_action_t) / action_range).pow(2).mean(dim=(1, 2))
        action_smooth = 0.5 * seq_smooth + 0.5 * first_smooth
        w = self.cost_weights
        if self.objective in {"hanwam_e040_phase_envelope", "phase_envelope"}:
            upper_ref = ref.view(1, -1).expand_as(temp)
            target_zone_mask = torch.ones_like(times, dtype=torch.bool) if remaining_seconds is None else target_margin_mask
            upper_ref = torch.where(target_zone_mask.view(1, -1), target_t.expand_as(temp), upper_ref)
            upper_bound = upper_ref + float(self.comfort_band_c)
            lower_bound = target_t - float(self.comfort_band_c)
            thermal_upper_envelope = torch.clamp(temp - upper_bound, min=0.0).pow(2).mean(dim=1)
            thermal_lower_envelope = (
                torch.clamp(lower_bound - temp, min=0.0).pow(2).mean(dim=1)
                * float(self.thermal_envelope_lower_weight)
            )
            if self.upper_reserve_c is None:
                upper_reserve = torch.zeros_like(energy_sum)
            else:
                reserve_error = torch.clamp(temp - (target_t + float(self.upper_reserve_c)), min=0.0).pow(2)
                upper_reserve = masked_mean(reserve_error, target_zone_mask)
            terminal_invariant = torch.zeros_like(energy_sum)
            if bool(target_zone_mask.any().detach().cpu().item()) and temp.shape[1] > 1:
                tail_steps = max(1, int(round(float(self.terminal_guard_tail_seconds) / float(self.step_seconds))))
                tail_steps = min(tail_steps, temp.shape[1] - 1)
                projection_steps = max(
                    1,
                    int(round(float(self.terminal_guard_projection_seconds) / float(self.step_seconds))),
                )
                terminal_rate = (temp[:, -1] - temp[:, -1 - tail_steps]) / float(tail_steps)
                projected_upper = temp[:, -1] + torch.clamp(terminal_rate, min=0.0) * float(projection_steps)
                projected_lower = temp[:, -1] + torch.clamp(terminal_rate, max=0.0) * float(projection_steps)
                terminal_upper = torch.clamp(
                    projected_upper - (target_t + float(self.terminal_guard_band_c)),
                    min=0.0,
                ).pow(2)
                terminal_lower = torch.clamp(
                    (target_t - float(self.terminal_guard_band_c)) - projected_lower,
                    min=0.0,
                ).pow(2)
                terminal_invariant = terminal_upper + terminal_lower
            envelope_weight = float(w.get("thermal_envelope", 0.0))
            parts = {
                "thermal_upper_envelope": thermal_upper_envelope
                * float(w.get("thermal_upper_envelope", envelope_weight)),
                "thermal_lower_envelope": thermal_lower_envelope
                * float(w.get("thermal_lower_envelope", envelope_weight)),
                "upper_reserve": upper_reserve * float(w.get("upper_reserve", 0.0)),
                "terminal_invariant": terminal_invariant * float(w.get("terminal_invariant", 0.0)),
                "energy": energy_sum * float(w.get("energy", 2.0)),
                "action_smooth": action_smooth * float(w.get("action_smooth", 0.20)),
            }
            return sum(parts.values()), parts

        parts = {
            "comfort_band_violation": comfort_band_violation * float(w.get("comfort_band_violation", 0.0)),
            "deadline_band_violation": deadline_band_violation * float(w.get("deadline_band_violation", 0.0)),
            "target_margin_band_violation": target_margin_band_violation
            * float(w.get("target_margin_band_violation", 0.0)),
            "target_center": target_center * float(w.get("target_center", 0.0)),
            "temperature_rate": temperature_rate * float(w.get("temperature_rate", 0.0)),
            "energy": energy_sum * float(w.get("energy", 2.0)),
            "action_smooth": action_smooth * float(w.get("action_smooth", 0.20)),
        }
        return sum(parts.values()), parts

    def _evaluate_actions(
        self,
        observation: np.ndarray,
        target: float,
        remaining_seconds: float | None,
        actions: torch.Tensor,
        obs_history_blocks: np.ndarray | None,
        act_history_blocks: np.ndarray | None,
        current_action: np.ndarray,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        physical = self._rollout_physical(observation, current_action, actions, obs_history_blocks, act_history_blocks)
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
        return PlannerResult(action=action, best_sequence=nominal_seq, debug=debug, history=pd.DataFrame(history_rows))
