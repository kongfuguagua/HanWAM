from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

try:
    from servicer.api_servicer.config import load_api_service_config
    from servicer.api_servicer.errors import ServiceError
except ModuleNotFoundError as exc:  # pragma: no cover - optional service deps
    if exc.name in {"fastapi", "pydantic"}:
        raise unittest.SkipTest(f"service dependency missing: {exc.name}")
    raise


ROOT = Path(__file__).resolve().parents[2]
HOST_CONFIG = ROOT / "servicer/config/api_service_hanwam.yml"
DOCKER_CONFIG = ROOT / "servicer/config/api_service_hanwam_docker.yml"
E65_CONFIG = ROOT / "control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml"
E65_CHECKPOINT = ROOT / "control/HanWAM/checkpoints/hanwam_e065_lowfreq10_positive_eev_energy_v1_mode1.pt"


class ApiServiceConfigTest(unittest.TestCase):
    def test_loads_hanwam_config_and_resolves_paths(self) -> None:
        config = load_api_service_config(HOST_CONFIG)
        self.assertEqual(config.service.port, 24243)
        self.assertEqual(config.service.num_threads, 4)
        self.assertEqual(config.controller.controller_type, "hanwam_wm_mpc")
        self.assertEqual(config.logging.path, ROOT / "outputs/servicer/hanwam_service.log")
        self.assertEqual(config.controller.algorithm_config, E65_CONFIG)
        self.assertEqual(config.controller.checkpoint, E65_CHECKPOINT)

    def test_loads_docker_config_with_container_absolute_paths(self) -> None:
        config = load_api_service_config(DOCKER_CONFIG)
        self.assertEqual(config.logging.path, Path("/app/outputs/servicer/hanwam_service.log"))
        self.assertEqual(
            config.controller.algorithm_config,
            Path("/app/control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml"),
        )
        self.assertEqual(
            config.controller.checkpoint,
            Path("/app/control/HanWAM/checkpoints/hanwam_e065_lowfreq10_positive_eev_energy_v1_mode1.pt"),
        )

    def test_rejects_invalid_port(self) -> None:
        text = HOST_CONFIG.read_text(encoding="utf-8")
        text = text.replace("port: 24243", "port: 70000")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ServiceError):
                load_api_service_config(path)

    def test_rejects_missing_checkpoint_field(self) -> None:
        text = HOST_CONFIG.read_text(encoding="utf-8")
        text = text.replace(
            "  checkpoint: control/HanWAM/checkpoints/hanwam_e065_lowfreq10_positive_eev_energy_v1_mode1.pt\n",
            "",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.yml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ServiceError):
                load_api_service_config(path)


if __name__ == "__main__":
    unittest.main()
