"""Controller factory for one-method experiments."""
from __future__ import annotations

from typing import Any

from .baselines import FixedActionController


def build_controller(
    method: str,
    mode: int,
    config: dict,
    bounds: dict[str, tuple[float, float]],
    args: Any,
) -> Any:
    if method == "historical":
        return None
    if method == "fixed":
        return FixedActionController(mode=mode, config=config)
    if method == "pid":
        from control.PID.controller import PIDController

        return PIDController(mode=mode, config=config, action_bounds=bounds)
    if method == "hanwam":
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Train a HanWAM checkpoint first: {args.checkpoint}")
        from control.HanWAM.controller import HanWAMController

        return HanWAMController(
            args.checkpoint,
            action_bounds=bounds,
            planner_config=(config.get("method") or {}).get("planner") or {},
            mode=mode,
        )
    raise ValueError(f"Unsupported method: {method}")
