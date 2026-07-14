"""Base interface for API-servicer controller adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from servicer.api_servicer.schemas import PlanRequest, PlanResponse, ResetResponse


class ControllerAdapter(ABC):
    @property
    @abstractmethod
    def controller_type(self) -> str:
        """Return the configured controller type name."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return controller metadata for readiness and inspection."""

    @abstractmethod
    def plan(self, request: PlanRequest) -> PlanResponse:
        """Return the next control action."""

    @abstractmethod
    def reset(self) -> ResetResponse:
        """Clear controller runtime state."""
