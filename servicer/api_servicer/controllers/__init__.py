"""Controller adapter factory."""

from __future__ import annotations

from servicer.api_servicer.config import ApiServiceConfig
from servicer.api_servicer.errors import ServiceError

from .base import ControllerAdapter
from .hanwam_wm_mpc import HanWAMWMMPCController


def build_controller(config: ApiServiceConfig) -> ControllerAdapter:
    controller_type = config.controller.controller_type
    if controller_type == "hanwam_wm_mpc":
        return HanWAMWMMPCController(config)
    raise ServiceError(
        "CONFIG_ERROR",
        f"unsupported controller.type: {controller_type}",
        status_code=500,
        details={"controller_type": controller_type},
    )
