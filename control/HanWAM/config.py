"""HanWAM paths and default training configuration."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "checkpoints" / "hanwam_mode1.pt"
