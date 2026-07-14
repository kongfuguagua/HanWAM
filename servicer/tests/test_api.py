from __future__ import annotations

import json
from pathlib import Path
import unittest

try:
    from fastapi.testclient import TestClient
    from servicer.api_servicer.app_factory import create_app
    from servicer.api_servicer.config import (
        ApiServiceConfig,
        ControllerRuntimeConfig,
        ServiceLoggingConfig,
        ServiceRuntimeConfig,
    )
    from servicer.api_servicer.controllers.base import ControllerAdapter
    from servicer.api_servicer.errors import ServiceError
    from servicer.api_servicer.schemas import PlanRequest, PlanResponse, ResetResponse
except ModuleNotFoundError as exc:  # pragma: no cover - optional service deps
    if exc.name in {"fastapi", "pydantic", "httpx"}:
        raise unittest.SkipTest(f"service dependency missing: {exc.name}")
    raise


def _payload() -> dict:
    return json.loads(Path("servicer/stubs/sample_plan_request.json").read_text(encoding="utf-8"))


class _FakeController(ControllerAdapter):
    @property
    def controller_type(self) -> str:
        return "hanwam_wm_mpc"

    def metadata(self) -> dict:
        return {"history_steps": 48, "rollout_steps": 96, "model_id": "fake:e065"}

    def plan(self, request: PlanRequest) -> PlanResponse:
        if request.mode != 1:
            raise ServiceError("UNSUPPORTED_REQUEST", "only mode=1 is supported", status_code=400)
        if len(request.obs_history) < 36 or len(request.act_history) < 36:
            raise ServiceError("VALIDATION_ERROR", "history is too short", status_code=422)
        reused_cached_action = getattr(self, "_planned", False)
        self._planned = True
        return PlanResponse(
            request_id=request.request_id,
            unit_id=request.unit_id,
            controller_type=self.controller_type,
            model_id="fake:e065",
            action={"freq_target": 40.0, "eev": 165.0, "fan_out": 750.0},
            valid_for_seconds=5,
            debug={"reused_cached_action": reused_cached_action},
        )

    def reset(self) -> ResetResponse:
        self._planned = False
        return ResetResponse(controller_type=self.controller_type, sessions_cleared=1)


def _config() -> ApiServiceConfig:
    root = Path.cwd()
    return ApiServiceConfig(
        config_path=root / "servicer/config/api_service_hanwam.yml",
        project_root=root,
        service=ServiceRuntimeConfig(host="127.0.0.1", port=24243, num_threads=1, device="cpu"),
        logging=ServiceLoggingConfig(
            level="info",
            mode="console",
            path=None,
            payloads=False,
            max_bytes=1024,
            backup_count=1,
        ),
        controller=ControllerRuntimeConfig(
            controller_type="hanwam_wm_mpc",
            mode=1,
            step_seconds=5,
            algorithm_config=root / "control/HanWAM/config/hanwam_e065_lowfreq10_positive_eev_energy_v1.yml",
            checkpoint=root / "control/HanWAM/checkpoints/hanwam_e065_lowfreq10_positive_eev_energy_v1_mode1.pt",
            planner_overrides={},
        ),
    )


class ApiServicerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = _config()
        controller = _FakeController()
        app = create_app(config, controller)
        cls.client = TestClient(app)

    def test_healthz(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_readyz_and_metadata(self) -> None:
        ready = self.client.get("/readyz")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["controller_type"], "hanwam_wm_mpc")
        metadata = self.client.get("/v1/metadata")
        self.assertEqual(metadata.status_code, 200)
        payload = metadata.json()
        self.assertEqual(payload["controller"]["history_steps"], 48)
        self.assertEqual(payload["controller"]["rollout_steps"], 96)
        self.assertEqual(payload["service"]["port"], 24243)

    def test_plan(self) -> None:
        self.client.post("/v1/reset")
        response = self.client.post("/v1/plan", json=_payload())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["controller_type"], "hanwam_wm_mpc")
        self.assertGreaterEqual(payload["action"]["freq_target"], 0.0)
        self.assertLessEqual(payload["action"]["freq_target"], 80.0)
        self.assertFalse(payload["debug"]["reused_cached_action"])

    def test_second_plan_reuses_unit_session(self) -> None:
        self.client.post("/v1/reset")
        first = self.client.post("/v1/plan", json=_payload())
        second = self.client.post("/v1/plan", json=_payload())
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(first.json()["debug"]["reused_cached_action"])
        self.assertTrue(second.json()["debug"]["reused_cached_action"])

    def test_reset_endpoint_clears_sessions(self) -> None:
        self.client.post("/v1/plan", json=_payload())
        reset = self.client.post("/v1/reset")
        self.assertEqual(reset.status_code, 200)
        payload = reset.json()
        self.assertEqual(payload["status"], "ok")
        self.assertGreaterEqual(payload["sessions_cleared"], 1)

    def test_short_history_returns_validation_error(self) -> None:
        payload = _payload()
        payload["obs_history"] = payload["obs_history"][:-1]
        response = self.client.post("/v1/plan", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "VALIDATION_ERROR")

    def test_missing_unit_id_returns_validation_error(self) -> None:
        payload = _payload()
        payload.pop("unit_id")
        response = self.client.post("/v1/plan", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "VALIDATION_ERROR")

    def test_blank_unit_id_returns_validation_error(self) -> None:
        payload = _payload()
        payload["unit_id"] = " "
        response = self.client.post("/v1/plan", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "VALIDATION_ERROR")

    def test_unsupported_mode_returns_explicit_error(self) -> None:
        payload = _payload()
        payload["mode"] = 3
        response = self.client.post("/v1/plan", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error_code"], "UNSUPPORTED_REQUEST")


if __name__ == "__main__":
    unittest.main()
