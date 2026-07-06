"""Base controller contracts used by all control methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ControllerContext:
    mode: int | str
    step_seconds: int = 5
    horizon_seconds: int | None = None
    elapsed_seconds: float = 0.0
    action_bounds: dict[str, tuple[float, float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlPlan:
    action: np.ndarray
    target_plan: pd.DataFrame = field(default_factory=pd.DataFrame)
    debug: dict[str, Any] = field(default_factory=dict)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.action = np.asarray(self.action, dtype=np.float32)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_plan": self.target_plan,
            "debug": self.debug,
            "history": self.history,
        }


class BaseController(ABC):
    policy_name = "base"

    def reset(self) -> None:
        """Reset stateful controller internals before a new run."""

    @abstractmethod
    def plan(
        self,
        observation: dict | np.ndarray,
        target: float,
        context: ControllerContext | None = None,
    ) -> ControlPlan:
        """Return the next target action [freq_target, eev, fan_out]."""

