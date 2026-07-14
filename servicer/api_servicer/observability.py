"""JSON-safe structured event logging helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import logging
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert common service/model values into strict JSON values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    if isinstance(value, BaseException):
        return str(value)
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Write one machine-readable JSON event without affecting service flow."""
    try:
        payload = {"event": str(event), **fields}
        message = json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except Exception as exc:  # pragma: no cover - logging must never break inference
        message = json.dumps(
            {"event": str(event), "logging_error": str(exc)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    logger.log(level, message)
