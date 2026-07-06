"""Power and electric-energy prediction for AC simulators."""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_DIR / "simu" / "energy" / "per_mode_models.pkl"
MODE_TO_NAME = {1: "制冷", 3: "制热"}


def mode_name(mode: int | str) -> str:
    if isinstance(mode, str):
        return mode
    return MODE_TO_NAME.get(int(mode), str(mode))


class EnergyModel:
    """Predict compressor-side power from actual controls.

    Inputs are always actual actuator values: [freq, eev, fan_out].

    When ``low_freq_correction`` is enabled, the model attenuates predictions
    in the sustained low-frequency band (default 10–18 Hz).  By default the
    correction uses an internal exponential moving average of ``freq`` so the
    model can be called one step at a time.  Callers that already have a
    smoothed frequency signal (or want to process a whole batch) can still
    pass ``freq_for_correction`` explicitly.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        mode: int | str = "制冷",
        low_freq_correction: bool = False,
        low_freq_band_hz: tuple[float, float] = (10.0, 18.0),
        low_freq_scale: float = 0.72,
        low_freq_ema_tau_seconds: float = 60.0,
        step_seconds: float = 5.0,
    ):
        self.models = joblib.load(model_path)
        self.mode = mode_name(mode)
        if self.mode not in self.models:
            raise KeyError(f"Energy model mode {self.mode!r} not found. Available: {list(self.models)}")
        self.model = self.models[self.mode]["gb"]
        self.low_freq_correction = bool(low_freq_correction)
        self.low_freq_band_hz = (float(low_freq_band_hz[0]), float(low_freq_band_hz[1]))
        self.low_freq_scale = float(low_freq_scale)
        self.low_freq_ema_tau_seconds = float(low_freq_ema_tau_seconds)
        self.step_seconds = float(step_seconds)
        if self.low_freq_ema_tau_seconds > 0:
            self._low_freq_ema_alpha = 1.0 - math.exp(
                -self.step_seconds / self.low_freq_ema_tau_seconds
            )
        else:
            self._low_freq_ema_alpha = 0.0
        self._freq_ema: float | None = None

    def reset(self) -> None:
        """Reset the internal frequency EMA state."""
        self._freq_ema = None

    def _low_freq_scale(self, freq: np.ndarray) -> np.ndarray:
        """Return a per-sample scaling factor that attenuates GB predictions at low freq.

        The GB model systematically overestimates power when the compressor is
        running in the sustained low-frequency band (roughly 10–18 Hz).  Outside
        this band the prediction is left unchanged; inside the band the prediction
        is attenuated linearly from ``low_freq_scale`` at the lower edge to 1.0 at
        the upper edge.

        Callers that want a long-term (smoothed) correction can pass an EMA of
        ``freq`` instead of the instantaneous value.
        """
        f = np.asarray(freq, dtype=float)
        lo, hi = self.low_freq_band_hz
        s = self.low_freq_scale
        return np.where(
            f <= lo,
            1.0,
            np.where(f >= hi, 1.0, s + (1.0 - s) * (f - lo) / (hi - lo)),
        )

    def _ema(self, freq: np.ndarray) -> np.ndarray:
        """Update and return the internal EMA for each frequency sample."""
        f = np.asarray(freq, dtype=float)
        out = np.empty_like(f, dtype=float)
        for i, value in enumerate(f):
            if self._freq_ema is None:
                self._freq_ema = float(value)
            elif self._low_freq_ema_alpha > 0.0:
                self._freq_ema += self._low_freq_ema_alpha * (value - self._freq_ema)
            else:
                self._freq_ema = float(value)
            out[i] = self._freq_ema
        return out

    def predict_power(
        self,
        controls: np.ndarray,
        freq_for_correction: np.ndarray | None = None,
    ) -> np.ndarray:
        controls = np.asarray(controls, dtype=np.float32)
        if controls.ndim == 1:
            controls = controls[None]
        power = self.model.predict(controls).astype(np.float32)
        power = np.where(controls[:, 0] <= 1.0, 0.0, power)
        if self.low_freq_correction:
            freq = (
                np.asarray(freq_for_correction, dtype=float)
                if freq_for_correction is not None
                else self._ema(controls[:, 0])
            )
            power = power * self._low_freq_scale(freq)
        return np.maximum(power, 0.0)

    def energy_kwh(self, controls: np.ndarray, step_seconds: float = 5.0) -> float:
        return float(self.predict_power(controls).sum() * step_seconds / 3600_000.0)
