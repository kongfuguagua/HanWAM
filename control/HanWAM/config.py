"""HanWAM paths and default training configuration."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "checkpoints" / "hanwam_mode1.pt"
DEFAULT_RESULTS = HERE / "results"

DEFAULT_LATENT_DIM = 64
DEFAULT_HIDDEN_DIM = 128
