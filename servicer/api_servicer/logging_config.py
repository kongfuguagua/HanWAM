"""Logging setup for the generic API servicer."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import ServiceLoggingConfig


def setup_logging(config: ServiceLoggingConfig) -> None:
    handlers: list[logging.Handler] = []
    if config.mode in {"console", "both"}:
        handlers.append(logging.StreamHandler())
    if config.mode in {"file", "both"}:
        if config.path is None:
            raise ValueError("logging.path is required for file logging")
        config.path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                config.path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
        )
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
