"""V4.1 following-oriented hybrid of slow physics and fast AR branches."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from simu.room.simulator import ContinuousEnthalpyRoomEnv

from .simulator import DEFAULT_V4_MODEL_PATH, HybridRoomV4Env


def _install_sklearn_16_loss_compatibility() -> None:
    """Allow the archived sklearn-1.6 fast branch to load under sklearn 1.9."""
    if "_loss" not in sys.modules:
        import sklearn._loss.loss as sklearn_loss
        sys.modules["_loss"] = sklearn_loss


class HybridRoomV4FollowingEnv:
    """Blend V4 long-horizon physics with a target-free fast AR response.

    Both branches receive the same initialization and, after reset, exactly
    the same three controls.  The current fast branch payload is the V3
    autoregressive plant model; its state columns exclude ``T_set``.
    """

    def __init__(
        self,
        slow_model_path: str | Path = DEFAULT_V4_MODEL_PATH,
        fast_model_path: str | Path | None = None,
        fast_weight: float = 0.5,
    ):
        if not 0.0 <= fast_weight <= 1.0:
            raise ValueError("fast_weight 必须在 [0,1] 范围内")
        _install_sklearn_16_loss_compatibility()
        self.slow = HybridRoomV4Env(slow_model_path)
        from sklearn.exceptions import InconsistentVersionWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            self.fast = (
                ContinuousEnthalpyRoomEnv()
                if fast_model_path is None
                else ContinuousEnthalpyRoomEnv(fast_model_path)
            )
        if not self.fast.feature_variant.startswith("autoregressive"):
            raise ValueError("快速支路必须是不使用目标温度的 autoregressive plant 模型")
        if "T_set" in self.fast.state_columns:
            raise ValueError("快速支路包含 T_set，拒绝加载")
        self.fast_weight = float(fast_weight)
        self._ready = False

    def reset(
        self,
        initial_observation: Mapping[str, float] | pd.Series | None = None,
        **initial_values: float,
    ) -> dict:
        observation = dict(initial_observation) if initial_observation is not None else {}
        observation.update(initial_values)
        slow = self.slow.reset(observation)
        fast_observation = dict(observation)
        # dataset_full contains valid cooling runs with a short mode=4 startup
        # prefix.  The V4.1 fast branch is an archived cooling-only V3 plant
        # model, so make its reset compatible with those startup rows while
        # keeping every physical/init feature and all runtime controls unchanged.
        fast_observation["mode"] = 1
        fast = self.fast.reset(initial_observation=fast_observation)
        self._ready = True
        return self._observation(slow, fast)

    def step(self, freq: float, eev: float, fan_out: float) -> dict:
        if not self._ready:
            raise RuntimeError("请先调用 reset()")
        slow = self.slow.step(freq, eev, fan_out)
        fast = self.fast.step(freq, eev, fan_out)
        return self._observation(slow, fast)

    def _observation(self, slow: dict, fast: dict) -> dict:
        temperature = (
            (1.0 - self.fast_weight) * float(slow["T_in"])
            + self.fast_weight * float(fast["T_in"])
        )
        return {
            "elapsed_seconds": float(slow["elapsed_seconds"]),
            "T_in": temperature,
            "T_in_slow": float(slow["T_in"]),
            "T_in_fast": float(fast["T_in"]),
            "T_out": float(slow["T_out"]),
            "fast_weight": self.fast_weight,
        }

    def simulate(self, initial_observation, controls) -> pd.DataFrame:
        rows = [self.reset(initial_observation)]
        for freq, eev, fan_out in np.asarray(controls, np.float32):
            rows.append(self.step(float(freq), float(eev), float(fan_out)))
        return pd.DataFrame(rows)
