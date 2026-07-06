"""Optional SwanLab experiment tracking."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class SwanLabTracker:
    def __init__(
        self,
        enabled: bool,
        project: str = "haier-jk-control",
        experiment_name: str | None = None,
        config: dict | None = None,
    ):
        self.enabled = enabled
        self.module = None
        self.run = None
        if not enabled:
            return
        try:
            import swanlab
        except Exception as exc:  # pragma: no cover - exercised through mock tests
            raise RuntimeError(
                "SwanLab is enabled but not available. Install with `pip install swanlab` "
                "and run `swanlab login` before using --swanlab."
            ) from exc
        self.module = swanlab
        self.run = swanlab.init(project=project, experiment_name=experiment_name, config=config or {})

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        if self.run is not None and hasattr(self.run, "log"):
            if step is None:
                self.run.log(payload)
            else:
                self.run.log(payload, step=step)
        elif self.module is not None:
            if step is None:
                self.module.log(payload)
            else:
                self.module.log(payload, step=step)

    def log_image(self, key: str, path: str | Path, caption: str | None = None) -> None:
        if not self.enabled:
            return
        assert self.module is not None
        image = self.module.Image(str(path), caption=caption or Path(path).name)
        self.log({key: image})

    def finish(self) -> None:
        if not self.enabled or self.module is None:
            return
        finish = getattr(self.module, "finish", None)
        if callable(finish):
            finish()

