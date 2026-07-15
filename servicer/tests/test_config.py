from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

try:
    from servicer.api_servicer.config import load_api_service_config, project_root
    from servicer.api_servicer.errors import ServiceError
except ModuleNotFoundError as exc:  # pragma: no cover - optional service deps
    if exc.name in {"fastapi", "pydantic"}:
        raise unittest.SkipTest(f"service dependency missing: {exc.name}")
    raise


HOST_CONFIG = Path("servicer/config/api_service_hanwam.yml")
DOCKER_CONFIG = Path("servicer/config/api_service_hanwam_docker.yml")


class ApiServiceConfigTest(unittest.TestCase):
    def test_loads_hanwam_config_and_resolves_paths(self) -> None:
        root = project_root()
        config = load_api_service_config(HOST_CONFIG)
        self.assertEqual(config.service.port, 8080)
        self.assertEqual(config.service.num_threads, 4)
        self.assertEqual(config.controller.controller_type, "hanwam_wm_mpc")
        self.assertEqual(config.logging.path, root / "outputs/servicer/hanwam_service.log")
        self.assertEqual(config.controller.algorithm_config, root / "control/HanWAM/config/hanwam.yml")
        self.assertEqual(
            config.controller.checkpoint,
            root / "control/HanWAM/checkpoints/hanwam_mode1.pt",
        )

    def test_loads_docker_config_with_container_absolute_paths(self) -> None:
        config = load_api_service_config(DOCKER_CONFIG)
        self.assertEqual(config.logging.path, Path("/app/outputs/servicer/hanwam_service.log"))
        self.assertEqual(config.controller.algorithm_config, Path("/app/control/HanWAM/config/hanwam.yml"))
        self.assertEqual(
            config.controller.checkpoint,
            Path("/app/control/HanWAM/checkpoints/hanwam_mode1.pt"),
        )

    def test_rejects_invalid_port(self) -> None:
        text = HOST_CONFIG.read_text(encoding="utf-8")
        text = text.replace("port: 8080", "port: 70000")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ServiceError):
                load_api_service_config(path)

    def test_rejects_missing_checkpoint_field(self) -> None:
        text = HOST_CONFIG.read_text(encoding="utf-8")
        text = text.replace("  checkpoint: control/HanWAM/checkpoints/hanwam_mode1.pt\n", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.yml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ServiceError):
                load_api_service_config(path)


if __name__ == "__main__":
    unittest.main()
