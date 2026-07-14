from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from servicer.api_servicer.app_factory import create_app
from servicer.api_servicer.config import (
    ApiServiceConfig,
    ControllerRuntimeConfig,
    ServiceLoggingConfig,
    ServiceRuntimeConfig,
)
from servicer.api_servicer.controllers.base import ControllerAdapter
from servicer.api_servicer.schemas import PlanRequest, PlanResponse, ResetResponse


class _FakeController(ControllerAdapter):
    @property
    def controller_type(self) -> str:
        return "fake"

    def metadata(self) -> dict:
        return {"model_id": "fake:model"}

    def plan(self, request: PlanRequest) -> PlanResponse:
        return PlanResponse(
            request_id=request.request_id,
            unit_id=request.unit_id,
            controller_type=self.controller_type,
            model_id="fake:model",
            action={"freq_target": 40.0, "eev": 165.0, "fan_out": 750.0},
            valid_for_seconds=5,
            debug={"planner": "fake"},
        )

    def reset(self) -> ResetResponse:
        return ResetResponse(controller_type=self.controller_type, sessions_cleared=0)


def _config(*, payloads: bool = True) -> ApiServiceConfig:
    root = Path.cwd()
    return ApiServiceConfig(
        config_path=root / "servicer/config/api_service_hanwam.yml",
        project_root=root,
        service=ServiceRuntimeConfig(host="127.0.0.1", port=8080, num_threads=1, device="cpu"),
        logging=ServiceLoggingConfig(
            level="info",
            mode="console",
            path=None,
            payloads=payloads,
            max_bytes=1024,
            backup_count=1,
        ),
        controller=ControllerRuntimeConfig(
            controller_type="fake",
            mode=1,
            step_seconds=5,
            algorithm_config=root / "algorithm.yml",
            checkpoint=root / "checkpoint.pt",
            planner_overrides={},
        ),
    )


def _payload(*, return_debug: bool) -> dict:
    return {
        "request_id": "debug-001",
        "unit_id": "ac-debug",
        "controller_type": "fake",
        "mode": 1,
        "step_seconds": 5,
        "target_temperature_c": 27.0,
        "obs_history": [{"T_in": 28.0}],
        "act_history": [{"freq_target": 40.0}],
        "return_debug": return_debug,
    }


class PlanDebugTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(_config(), _FakeController()))

    def test_return_debug_echoes_raw_and_validated_request(self) -> None:
        request_body = json.dumps(_payload(return_debug=True), separators=(",", ":")).encode("utf-8")
        response = self.client.post("/v1/plan", content=request_body, headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 200)
        debug = response.json()["debug"]
        self.assertEqual(debug["request_body"], _payload(return_debug=True))
        self.assertEqual(debug["validated_request"]["elapsed_seconds"], 0.0)
        self.assertEqual(debug["request_body_sha256"], hashlib.sha256(request_body).hexdigest())
        self.assertEqual(debug["request_body_bytes"], len(request_body))

    def test_request_body_is_not_echoed_without_debug_flag(self) -> None:
        response = self.client.post("/v1/plan", json=_payload(return_debug=False))
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("request_body", response.json()["debug"])

    def test_validation_error_echoes_body_when_debug_is_requested(self) -> None:
        payload = _payload(return_debug=True)
        payload.pop("unit_id")
        response = self.client.post("/v1/plan", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["details"]["request_body"], payload)

    def test_plan_payload_events_are_logged(self) -> None:
        with self.assertLogs("servicer.api", level="INFO") as captured:
            response = self.client.post("/v1/plan", json=_payload(return_debug=False))
        self.assertEqual(response.status_code, 200)
        output = "\n".join(captured.output)
        self.assertIn('"event":"plan.request"', output)
        self.assertIn('"event":"plan.response"', output)
        self.assertIn('"request_id":"debug-001"', output)


if __name__ == "__main__":
    unittest.main()
