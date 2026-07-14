"""HanWAM WM+MPPI controller adapter for the generic API servicer."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
import logging
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch
import yaml

from control.HanWAM.dataloader import Normalizer
from control.HanWAM.model import build_wam_model
from control.HanWAM.planner import MPPIPlanner
from control.HanWAM.type import WAM_ACTION_COLS, WAM_OBS_COLS, WAM_PHYSICAL_COLS
from control.HanWAM.utils import cooling_ddl_seconds

from servicer.api_servicer.config import ApiServiceConfig
from servicer.api_servicer.controllers.base import ControllerAdapter
from servicer.api_servicer.errors import ServiceError
from servicer.api_servicer.observability import log_event
from servicer.api_servicer.schemas import PlanRequest, PlanResponse, ResetResponse


LOGGER = logging.getLogger("servicer.algorithm.hanwam")


@dataclass(frozen=True)
class HanWAMMetadata:
    controller_type: str
    model_id: str
    algorithm_config: str
    checkpoint: str
    checkpoint_sha256: str
    obs_schema: list[str]
    action_schema: list[str]
    physical_schema: list[str]
    action_bounds: dict[str, tuple[float, float]]
    mode: int
    step_seconds: int
    device: str
    history_steps: int
    frames_per_block: int
    history_blocks: int
    future_blocks: int
    rollout_steps: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "controller_type": self.controller_type,
            "model_id": self.model_id,
            "algorithm_config": self.algorithm_config,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "obs_schema": self.obs_schema,
            "action_schema": self.action_schema,
            "physical_schema": self.physical_schema,
            "action_bounds": {key: list(value) for key, value in self.action_bounds.items()},
            "mode": self.mode,
            "step_seconds": self.step_seconds,
            "device": self.device,
            "history_steps": self.history_steps,
            "frames_per_block": self.frames_per_block,
            "history_blocks": self.history_blocks,
            "future_blocks": self.future_blocks,
            "rollout_steps": self.rollout_steps,
        }


@dataclass
class _HanWAMUnitSession:
    planner: MPPIPlanner
    cached_actions: list[np.ndarray] = field(default_factory=list)
    ddl_seconds: float | None = None
    ddl_target_temperature_c: float | None = None
    ddl_start_t_in_c: float | None = None
    ddl_start_t_out_c: float | None = None
    request_count: int = 0
    plan_count: int = 0
    cache_hit_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ServiceError(
            "CONFIG_ERROR",
            f"algorithm config file not found: {path}",
            status_code=500,
            details={"algorithm_config": str(path)},
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ServiceError("CONFIG_ERROR", "algorithm config must be a mapping", status_code=500)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(raw: str) -> torch.device:
    value = str(raw).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ServiceError("SERVICE_NOT_READY", "CUDA requested but not available", status_code=503)
    return torch.device(value)


def _json_float(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class HanWAMWMMPCController(ControllerAdapter):
    def __init__(self, config: ApiServiceConfig):
        self.config = config
        self._controller_type = str(config.controller.controller_type)
        self.mode = int(config.controller.mode)
        self.step_seconds = int(config.controller.step_seconds)
        if self.step_seconds != 5:
            raise ServiceError(
                "CONFIG_ERROR",
                "hanwam_wm_mpc currently requires controller.step_seconds=5",
                status_code=500,
                details={"step_seconds": self.step_seconds},
            )
        if self.mode != 1:
            raise ServiceError(
                "CONFIG_ERROR",
                "hanwam_wm_mpc currently supports mode=1 only",
                status_code=500,
                details={"mode": self.mode},
            )
        if config.service.device == "cpu":
            torch.set_num_threads(int(config.service.num_threads))
        self.algorithm_config = _load_yaml(config.controller.algorithm_config)
        if not config.controller.checkpoint.exists():
            raise ServiceError(
                "SERVICE_NOT_READY",
                f"checkpoint file not found: {config.controller.checkpoint}",
                status_code=503,
                details={"checkpoint": str(config.controller.checkpoint)},
            )
        checkpoint = torch.load(config.controller.checkpoint, map_location="cpu", weights_only=False)
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = _sha256(config.controller.checkpoint)
        self.obs_cols = list(checkpoint["obs_cols"])
        self.action_cols = list(checkpoint["target_action_cols"])
        self.physical_cols = list(checkpoint["physical_cols"])
        self._validate_checkpoint_schema()
        self.device = _resolve_device(config.service.device)
        self.obs_norm = Normalizer.from_dict(checkpoint["obs_norm"])
        self.action_norm = Normalizer.from_dict(checkpoint["target_action_norm"])
        self.physical_norm = Normalizer.from_dict(checkpoint["physical_norm"])
        self.model = build_wam_model(checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.frames_per_block = int(checkpoint.get("frames_per_block", getattr(self.model, "frames_per_block", 12)))
        self.history_blocks = int(checkpoint.get("history_blocks", getattr(self.model, "history_blocks", 3)))
        self.future_blocks = int(checkpoint.get("future_blocks", getattr(self.model, "future_blocks", 5)))
        self.history_steps = self.frames_per_block * self.history_blocks
        self.rollout_steps = self.frames_per_block * self.future_blocks
        self.action_bounds = self._action_bounds_from_algorithm_config()
        self.planner_config = self._merged_planner_config()
        self._sessions: dict[str, _HanWAMUnitSession] = {}
        self._sessions_lock = threading.Lock()
        experiment_name = str((self.algorithm_config.get("experiment") or {}).get("name") or self._controller_type)
        self.model_id = f"{experiment_name}:{self.checkpoint_sha256[:12]}"
        self._metadata = HanWAMMetadata(
            controller_type=self._controller_type,
            model_id=self.model_id,
            algorithm_config=str(config.controller.algorithm_config),
            checkpoint=str(config.controller.checkpoint),
            checkpoint_sha256=self.checkpoint_sha256,
            obs_schema=list(WAM_OBS_COLS),
            action_schema=list(WAM_ACTION_COLS),
            physical_schema=list(WAM_PHYSICAL_COLS),
            action_bounds=self.action_bounds,
            mode=self.mode,
            step_seconds=self.step_seconds,
            device=str(self.device),
            history_steps=self.history_steps,
            frames_per_block=self.frames_per_block,
            history_blocks=self.history_blocks,
            future_blocks=self.future_blocks,
            rollout_steps=self.rollout_steps,
        )
        log_event(LOGGER, "algorithm.ready", metadata=self._metadata.as_dict(), planner_config=self.planner_config)

    @property
    def controller_type(self) -> str:
        return self._controller_type

    def metadata(self) -> dict[str, Any]:
        payload = self._metadata.as_dict()
        with self._sessions_lock:
            payload["active_sessions"] = len(self._sessions)
        payload["session_mode"] = "unit_stateful"
        return payload

    def reset(self) -> ResetResponse:
        with self._sessions_lock:
            sessions_cleared = len(self._sessions)
            self._sessions.clear()
        response = ResetResponse(
            controller_type=self.controller_type,
            sessions_cleared=sessions_cleared,
        )
        log_event(LOGGER, "algorithm.sessions_reset", sessions_cleared=sessions_cleared)
        return response

    def _validate_checkpoint_schema(self) -> None:
        mismatches = {}
        if self.obs_cols != list(WAM_OBS_COLS):
            mismatches["obs_cols"] = {"expected": list(WAM_OBS_COLS), "actual": self.obs_cols}
        if self.action_cols != list(WAM_ACTION_COLS):
            mismatches["action_cols"] = {"expected": list(WAM_ACTION_COLS), "actual": self.action_cols}
        if self.physical_cols != list(WAM_PHYSICAL_COLS):
            mismatches["physical_cols"] = {"expected": list(WAM_PHYSICAL_COLS), "actual": self.physical_cols}
        if mismatches:
            raise ServiceError(
                "CONFIG_ERROR",
                "checkpoint schema does not match HanWAM schema",
                status_code=500,
                details=mismatches,
            )

    def _action_bounds_from_algorithm_config(self) -> dict[str, tuple[float, float]]:
        spaces = ((self.algorithm_config.get("method") or {}).get("action_space_by_mode") or {})
        payload = spaces.get(str(self.mode))
        if payload is None:
            payload = self.checkpoint.get("target_action_bounds") or {}
        bounds = {}
        for col in WAM_ACTION_COLS:
            raw = payload.get(col)
            if raw is None or len(raw) != 2:
                raise ServiceError(
                    "CONFIG_ERROR",
                    f"missing action bound for {col}",
                    status_code=500,
                    details={"column": col},
                )
            bounds[col] = (float(raw[0]), float(raw[1]))
        return bounds

    def _merged_planner_config(self) -> dict[str, Any]:
        base = dict(self.checkpoint.get("planner_config") or {})
        algorithm_planner = dict(((self.algorithm_config.get("method") or {}).get("planner") or {}))
        service_overrides = dict(self.config.controller.planner_overrides)
        merged = {**base, **algorithm_planner, **service_overrides}
        cost_weights = dict(base.get("cost_weights") or {})
        cost_weights.update(algorithm_planner.get("cost_weights") or {})
        cost_weights.update(service_overrides.get("cost_weights") or {})
        if cost_weights:
            merged["cost_weights"] = cost_weights
        merged["algorithm"] = "mppi"
        merged.setdefault("frames_per_block", self.frames_per_block)
        merged.setdefault("history_blocks", self.history_blocks)
        merged.setdefault("future_blocks", self.future_blocks)
        merged.setdefault("step_seconds", self.step_seconds)
        merged.setdefault("horizon_steps", self.rollout_steps)
        merged["mode"] = self.mode
        return merged

    def _new_planner(self) -> MPPIPlanner:
        return MPPIPlanner(
            model=self.model,
            obs_norm=self.obs_norm,
            action_norm=self.action_norm,
            physical_norm=self.physical_norm,
            obs_cols=self.obs_cols,
            physical_cols=self.physical_cols,
            action_bounds=self.action_bounds,
            config=self.planner_config,
            device=self.device,
        )

    def _session_for_unit(self, unit_id: str) -> _HanWAMUnitSession:
        with self._sessions_lock:
            session = self._sessions.get(unit_id)
            if session is None:
                session = _HanWAMUnitSession(planner=self._new_planner())
                self._sessions[unit_id] = session
            return session

    def _validate_request(self, request: PlanRequest) -> None:
        if request.controller_type is not None and request.controller_type != self.controller_type:
            raise ServiceError(
                "UNSUPPORTED_REQUEST",
                "request controller_type does not match configured controller",
                status_code=400,
                request_id=request.request_id,
                details={"request_controller_type": request.controller_type, "configured_controller_type": self.controller_type},
            )
        if int(request.mode) != self.mode:
            raise ServiceError(
                "UNSUPPORTED_REQUEST",
                "request mode does not match configured controller mode",
                status_code=400,
                request_id=request.request_id,
                details={"request_mode": request.mode, "configured_mode": self.mode},
            )
        if int(request.step_seconds) != self.step_seconds:
            raise ServiceError(
                "UNSUPPORTED_REQUEST",
                "request step_seconds does not match configured controller step_seconds",
                status_code=400,
                request_id=request.request_id,
                details={"request_step_seconds": request.step_seconds, "configured_step_seconds": self.step_seconds},
            )
        if request.unit_id is None or not str(request.unit_id).strip():
            raise ServiceError(
                "VALIDATION_ERROR",
                "unit_id is required for stateful control sessions",
                status_code=422,
                request_id=request.request_id,
                details={"field": "unit_id"},
            )
        if len(request.obs_history) != self.history_steps:
            raise ServiceError(
                "VALIDATION_ERROR",
                f"obs_history must contain exactly {self.history_steps} frames",
                status_code=422,
                request_id=request.request_id,
                details={"expected": self.history_steps, "actual": len(request.obs_history)},
            )
        if len(request.act_history) != self.history_steps:
            raise ServiceError(
                "VALIDATION_ERROR",
                f"act_history must contain exactly {self.history_steps} frames",
                status_code=422,
                request_id=request.request_id,
                details={"expected": self.history_steps, "actual": len(request.act_history)},
            )
        missing_obs = sorted({col for col in WAM_OBS_COLS for row in request.obs_history if col not in row})
        if missing_obs:
            raise ServiceError(
                "VALIDATION_ERROR",
                "obs_history is missing required fields",
                status_code=422,
                request_id=request.request_id,
                details={"missing": missing_obs},
            )
        missing_act = sorted({col for col in WAM_ACTION_COLS for row in request.act_history if col not in row})
        if missing_act:
            raise ServiceError(
                "VALIDATION_ERROR",
                "act_history is missing required fields",
                status_code=422,
                request_id=request.request_id,
                details={"missing": missing_act},
            )

    def _arrays_from_request(self, request: PlanRequest) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(
            [[float(row[col]) for col in WAM_OBS_COLS] for row in request.obs_history],
            dtype=np.float32,
        )
        act = np.asarray(
            [[float(row[col]) for col in WAM_ACTION_COLS] for row in request.act_history],
            dtype=np.float32,
        )
        obs_blocks = obs.reshape(self.history_blocks, self.frames_per_block, len(WAM_OBS_COLS))
        act_blocks = act.reshape(self.history_blocks, self.frames_per_block, len(WAM_ACTION_COLS))
        return obs_blocks, act_blocks

    def _refresh_session_deadline(self, session: _HanWAMUnitSession, request: PlanRequest) -> bool:
        target = float(request.target_temperature_c)
        target_changed = (
            session.ddl_target_temperature_c is not None
            and not np.isclose(session.ddl_target_temperature_c, target, rtol=0.0, atol=1e-6)
        )
        if session.ddl_seconds is not None and session.ddl_target_temperature_c is not None and not target_changed:
            return False

        first_obs = request.obs_history[0]
        start_t_in = float(first_obs["T_in"])
        start_t_out = float(first_obs["T_out"])
        session.ddl_seconds = cooling_ddl_seconds(start_t_in, start_t_out, target)
        session.ddl_target_temperature_c = target
        session.ddl_start_t_in_c = start_t_in
        session.ddl_start_t_out_c = start_t_out
        if target_changed:
            session.cached_actions = []
            session.planner.reset()
        return True

    @staticmethod
    def _deadline_debug(
        session: _HanWAMUnitSession,
        request: PlanRequest,
        deadline_recomputed: bool,
    ) -> dict[str, Any]:
        return {
            "effective_deadline_seconds": session.ddl_seconds,
            "deadline_source": "auto_first_frame",
            "deadline_recomputed": bool(deadline_recomputed),
            "deadline_start_t_in_c": session.ddl_start_t_in_c,
            "deadline_start_t_out_c": session.ddl_start_t_out_c,
            "request_deadline_seconds": request.deadline_seconds,
        }

    def plan(self, request: PlanRequest) -> PlanResponse:
        self._validate_request(request)
        obs_blocks, act_blocks = self._arrays_from_request(request)
        observation = obs_blocks.reshape(self.history_steps, len(WAM_OBS_COLS))[-1]
        current_action = act_blocks.reshape(self.history_steps, len(WAM_ACTION_COLS))[-1]
        unit_id = str(request.unit_id).strip()
        session = self._session_for_unit(unit_id)
        with session.lock:
            session.request_count += 1
            deadline_recomputed = self._refresh_session_deadline(session, request)
            remaining_seconds = max(0.0, float(session.ddl_seconds) - float(request.elapsed_seconds))
            deadline_debug = self._deadline_debug(session, request, deadline_recomputed)
            if self.config.logging.payloads:
                log_event(
                    LOGGER,
                    "algorithm.input",
                    request_id=request.request_id,
                    unit_id=unit_id,
                    model_id=self.model_id,
                    input={
                        "target_temperature_c": float(request.target_temperature_c),
                        "elapsed_seconds": float(request.elapsed_seconds),
                        "deadline_seconds": request.deadline_seconds,
                        "effective_deadline_seconds": session.ddl_seconds,
                        "deadline_source": deadline_debug["deadline_source"],
                        "deadline_recomputed": deadline_recomputed,
                        "remaining_seconds": remaining_seconds,
                        "observation_schema": self.obs_cols,
                        "action_schema": self.action_cols,
                        "observation": observation,
                        "current_action": current_action,
                        "obs_history_blocks": obs_blocks,
                        "act_history_blocks": act_blocks,
                    },
                    session={
                        "request_count": session.request_count,
                        "plan_count": session.plan_count,
                        "cache_hit_count": session.cache_hit_count,
                        "cached_actions_before": len(session.cached_actions),
                    },
                )
            if session.cached_actions:
                raw_action = np.asarray(session.cached_actions.pop(0), dtype=np.float32)
                action = np.clip(raw_action, session.planner._lo, session.planner._hi).astype(float)
                planned_sequence = None
                planner_debug = None
                session.cache_hit_count += 1
                debug = {
                    "planner": getattr(session.planner, "planner_name", "mppi"),
                    "plan_ms": 0.0,
                    "history_steps": self.history_steps,
                    "horizon_steps": int(session.planner.horizon_steps),
                    "rollout_steps": self.rollout_steps,
                    "stateful_session": True,
                    "reused_cached_action": True,
                    "cached_actions_remaining": len(session.cached_actions),
                    "session_request_count": session.request_count,
                    "session_plan_count": session.plan_count,
                    "session_cache_hit_count": session.cache_hit_count,
                }
                debug.update(deadline_debug)
            else:
                try:
                    with torch.inference_mode():
                        result = session.planner.plan(
                            observation,
                            target=float(request.target_temperature_c),
                            remaining_seconds=remaining_seconds,
                            obs_history_blocks=obs_blocks,
                            act_history_blocks=act_blocks,
                            current_action=current_action,
                        )
                except Exception as exc:
                    log_event(
                        LOGGER,
                        "algorithm.error",
                        level=logging.ERROR,
                        request_id=request.request_id,
                        unit_id=unit_id,
                        model_id=self.model_id,
                        error=exc,
                    )
                    raise
                session.plan_count += 1
                planned_sequence = result.best_sequence
                planner_debug = result.debug
                if session.planner.control_interval_steps > 1:
                    session.cached_actions = [
                        np.asarray(action, dtype=np.float32)
                        for action in result.best_sequence[1 : session.planner.control_interval_steps]
                    ]
                action = result.action.astype(float)
                debug = {
                    "planner": result.debug.get("hanwam_planner", "mppi"),
                    "plan_ms": float(result.debug.get("hanwam_plan_ms", 0.0)),
                    "history_steps": self.history_steps,
                    "horizon_steps": int(session.planner.horizon_steps),
                    "rollout_steps": self.rollout_steps,
                    "stateful_session": True,
                    "reused_cached_action": False,
                    "cached_actions_remaining": len(session.cached_actions),
                    "session_request_count": session.request_count,
                    "session_plan_count": session.plan_count,
                    "session_cache_hit_count": session.cache_hit_count,
                }
                debug.update(deadline_debug)
                if request.return_debug:
                    debug.update({key: _json_float(value) for key, value in result.debug.items()})
            action_payload = {
                "freq_target": float(action[0]),
                "eev": float(action[1]),
                "fan_out": float(action[2]),
            }
            if self.config.logging.payloads:
                log_event(
                    LOGGER,
                    "algorithm.output",
                    request_id=request.request_id,
                    unit_id=unit_id,
                    model_id=self.model_id,
                    source="cached_action" if debug["reused_cached_action"] else "planner",
                    action=action_payload,
                    planned_sequence=planned_sequence,
                    planner_debug=planner_debug,
                    debug=debug,
                )
        return PlanResponse(
            request_id=request.request_id,
            unit_id=unit_id,
            controller_type=self.controller_type,
            model_id=self.model_id,
            action=action_payload,
            valid_for_seconds=self.step_seconds,
            debug=debug,
        )
