"""Explicit YAML configuration for the generic API servicer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ServiceError


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_mapping(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ServiceError("CONFIG_ERROR", f"{name} must be a mapping", status_code=500)
    return payload


def _required(payload: dict[str, Any], name: str, section: str) -> Any:
    if name not in payload or payload[name] is None:
        raise ServiceError("CONFIG_ERROR", f"missing {section}.{name}", status_code=500)
    return payload[name]


def _resolve_path(raw: Any, root: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return root / path


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    host: str
    port: int
    num_threads: int
    device: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "num_threads": self.num_threads,
            "device": self.device,
        }


@dataclass(frozen=True)
class ServiceLoggingConfig:
    level: str
    mode: str
    path: Path | None
    payloads: bool
    max_bytes: int
    backup_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "mode": self.mode,
            "path": None if self.path is None else str(self.path),
            "payloads": self.payloads,
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count,
        }


@dataclass(frozen=True)
class ControllerRuntimeConfig:
    controller_type: str
    mode: int
    step_seconds: int
    algorithm_config: Path
    checkpoint: Path
    planner_overrides: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.controller_type,
            "mode": self.mode,
            "step_seconds": self.step_seconds,
            "algorithm_config": str(self.algorithm_config),
            "checkpoint": str(self.checkpoint),
            "planner_overrides": self.planner_overrides,
        }


@dataclass(frozen=True)
class ApiServiceConfig:
    config_path: Path
    project_root: Path
    service: ServiceRuntimeConfig
    logging: ServiceLoggingConfig
    controller: ControllerRuntimeConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "project_root": str(self.project_root),
            "service": self.service.as_dict(),
            "logging": self.logging.as_dict(),
            "controller": self.controller.as_dict(),
        }


def load_api_service_config(config_path: str | Path) -> ApiServiceConfig:
    root = project_root()
    path = _resolve_path(config_path, root)
    if not path.exists():
        raise ServiceError(
            "CONFIG_ERROR",
            f"service config file not found: {path}",
            status_code=500,
            details={"config_path": str(path)},
        )
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    payload = _require_mapping(raw, "service config")
    service_payload = _require_mapping(_required(payload, "service", "root"), "service")
    logging_payload = _require_mapping(_required(payload, "logging", "root"), "logging")
    controller_payload = _require_mapping(_required(payload, "controller", "root"), "controller")

    host = str(_required(service_payload, "host", "service"))
    port = int(_required(service_payload, "port", "service"))
    if port < 1 or port > 65535:
        raise ServiceError("CONFIG_ERROR", "service.port must be between 1 and 65535", status_code=500)
    num_threads = int(_required(service_payload, "num_threads", "service"))
    if num_threads < 1:
        raise ServiceError("CONFIG_ERROR", "service.num_threads must be >= 1", status_code=500)
    device = str(_required(service_payload, "device", "service"))

    log_level = str(_required(logging_payload, "level", "logging")).lower()
    if log_level not in {"debug", "info", "warning", "error", "critical"}:
        raise ServiceError("CONFIG_ERROR", "logging.level is invalid", status_code=500)
    log_mode = str(_required(logging_payload, "mode", "logging")).lower()
    if log_mode not in {"console", "file", "both"}:
        raise ServiceError("CONFIG_ERROR", "logging.mode must be console, file, or both", status_code=500)
    log_path = None
    if "path" in logging_payload and logging_payload["path"] is not None:
        log_path = _resolve_path(logging_payload["path"], root)
    if log_mode in {"file", "both"} and log_path is None:
        raise ServiceError("CONFIG_ERROR", "logging.path is required for file logging", status_code=500)
    log_payloads = logging_payload.get("payloads", False)
    if not isinstance(log_payloads, bool):
        raise ServiceError("CONFIG_ERROR", "logging.payloads must be a boolean", status_code=500)
    log_max_bytes = int(logging_payload.get("max_bytes", 50 * 1024 * 1024))
    if log_max_bytes < 1:
        raise ServiceError("CONFIG_ERROR", "logging.max_bytes must be >= 1", status_code=500)
    log_backup_count = int(logging_payload.get("backup_count", 5))
    if log_backup_count < 0:
        raise ServiceError("CONFIG_ERROR", "logging.backup_count must be >= 0", status_code=500)

    controller_type = str(_required(controller_payload, "type", "controller"))
    controller_mode = int(_required(controller_payload, "mode", "controller"))
    step_seconds = int(_required(controller_payload, "step_seconds", "controller"))
    algorithm_config = _resolve_path(_required(controller_payload, "algorithm_config", "controller"), root)
    checkpoint = _resolve_path(_required(controller_payload, "checkpoint", "controller"), root)
    planner_overrides = controller_payload.get("planner_overrides") or {}
    if not isinstance(planner_overrides, dict):
        raise ServiceError("CONFIG_ERROR", "controller.planner_overrides must be a mapping", status_code=500)

    service_config = ServiceRuntimeConfig(
        host=host,
        port=port,
        num_threads=num_threads,
        device=device,
    )
    logging_config = ServiceLoggingConfig(
        level=log_level,
        mode=log_mode,
        path=log_path,
        payloads=log_payloads,
        max_bytes=log_max_bytes,
        backup_count=log_backup_count,
    )
    controller_config = ControllerRuntimeConfig(
        controller_type=controller_type,
        mode=controller_mode,
        step_seconds=step_seconds,
        algorithm_config=algorithm_config,
        checkpoint=checkpoint,
        planner_overrides=dict(planner_overrides),
    )
    return ApiServiceConfig(
        config_path=path,
        project_root=root,
        service=service_config,
        logging=logging_config,
        controller=controller_config,
    )
