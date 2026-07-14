"""CLI entrypoint for the generic API servicer."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .app_factory import create_app
from .config import load_api_service_config
from .controllers import build_controller
from .logging_config import setup_logging
from .observability import log_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the generic control API servicer.")
    parser.add_argument("--config", required=True, help="Path to the api-servicer YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_api_service_config(args.config)
    setup_logging(config.logging)
    controller = build_controller(config)
    app = create_app(config, controller)
    log_event(
        logging.getLogger(__name__),
        "service.launching",
        config=config.as_dict(),
        controller=controller.metadata(),
    )
    uvicorn.run(
        app,
        host=config.service.host,
        port=config.service.port,
        log_level=config.logging.level,
    )


if __name__ == "__main__":
    main()
