"""Generic HTTP schemas shared by controller adapters."""

from __future__ import annotations

from typing import Any
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _finite(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _finite_mapping(payload: dict[str, Any], name: str) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    converted = {}
    for key, value in payload.items():
        converted[str(key)] = _finite(value, f"{name}.{key}")
    return converted


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    unit_id: str
    controller_type: str | None = None
    mode: int
    step_seconds: int
    target_temperature_c: float
    elapsed_seconds: float = 0.0
    deadline_seconds: float | None = None
    obs_history: list[dict[str, float]]
    act_history: list[dict[str, float]]
    return_debug: bool = False

    @field_validator("unit_id")
    @classmethod
    def _non_empty_unit_id(cls, value: str) -> str:
        unit_id = str(value).strip()
        if not unit_id:
            raise ValueError("unit_id must be non-empty")
        return unit_id

    @field_validator("target_temperature_c", "elapsed_seconds", "deadline_seconds")
    @classmethod
    def _finite_scalar(cls, value: float | None, info):
        if value is None:
            return None
        return _finite(value, info.field_name)

    @field_validator("obs_history")
    @classmethod
    def _finite_obs_history(cls, value: list[dict[str, Any]]) -> list[dict[str, float]]:
        return [_finite_mapping(row, f"obs_history[{idx}]") for idx, row in enumerate(value)]

    @field_validator("act_history")
    @classmethod
    def _finite_act_history(cls, value: list[dict[str, Any]]) -> list[dict[str, float]]:
        return [_finite_mapping(row, f"act_history[{idx}]") for idx, row in enumerate(value)]


class PlanResponse(BaseModel):
    status: str = "ok"
    request_id: str | None = None
    unit_id: str | None = None
    controller_type: str
    model_id: str
    action: dict[str, float]
    valid_for_seconds: int
    schema_version: str = "control.plan.v1"
    debug: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    controller_type: str
    model_id: str


class ResetResponse(BaseModel):
    status: str = "ok"
    controller_type: str
    sessions_cleared: int
    schema_version: str = "control.reset.v1"


class ErrorResponse(BaseModel):
    status: str = "error"
    request_id: str | None = None
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
