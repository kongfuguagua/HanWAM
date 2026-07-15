from __future__ import annotations

import json
from pathlib import Path
import unittest

try:
    from fastapi.testclient import TestClient
    from servicer.api_servicer.app_factory import create_app
    from servicer.api_servicer.config import load_api_service_config
    from servicer.api_servicer.controllers.base import ControllerAdapter
    from servicer.api_servicer.errors import ServiceError
    from servicer.api_servicer.schemas import PlanRequest, PlanResponse, ResetResponse
except ModuleNotFoundError as exc:  # pragma: no cover - optional service deps
    if exc.name in {"fastapi", "pydantic", "httpx"}:
        raise unittest.SkipTest(f"service dependency missing: {exc.name}")
    raise


def _payload() -> dict:
    payload = json.loads(Path("servicer/stubs/sample_plan_request.json").read_text(encoding="utf-8"))
    payload["return_debug"] = False
    while len(payload["obs_history"]) < 48:
        payload["obs_history"].append(dict(payload["obs_history"][-1]))
    while len(payload["act_history"]) < 48:
        payload["act_history"].append(dict(payload["act_history"][-1]))
    return payload


def _minimal_payload() -> dict:
    return {
        "request_id": "sample-001",
        "unit_id": "ac-demo",
        "controller_type": "hanwam_wm_mpc",
        "mode": 1,
        "step_seconds": 5,
        "target_temperature_c": 27.0,
        "obs_history": [{"T_in": 28.0}],
        "act_history": [{"freq_target": 40.0}],
        "return_debug": False,
    }


class _FakeHanWAMController(ControllerAdapter):
    def __init__(self) -> None:
        self.sessions: set[str] = set()

    @property
    def controller_type(self) -> str:
        return "hanwam_wm_mpc"

    def metadata(self) -> dict:
        return {
            "controller_type": self.controller_type,
            "model_id": "fake-hanwam:model",
            "history_steps": 48,
            "rollout_steps": 96,
            "active_sessions": len(self.sessions),
        }

    def plan(self, request: PlanRequest) -> PlanResponse:
        if request.controller_type not in {None, self.controller_type}:
            raise ServiceError(
                "UNSUPPORTED_REQUEST",
                "controller_type mismatch",
                status_code=400,
                request_id=request.request_id,
            )
        if request.mode != 1:
            raise ServiceError(
                "UNSUPPORTED_REQUEST",
                "mode is not supported",
                status_code=400,
                request_id=request.request_id,
            )
        if len(request.obs_history) < 48 or len(request.act_history) < 48:
            raise ServiceError(
                "VALIDATION_ERROR",
                "history is shorter than model requirement",
                status_code=422,
                request_id=request.request_id,
            )
        reused = request.unit_id in self.sessions
        self.sessions.add(request.unit_id)
        return PlanResponse(
            request_id=request.request_id,
            unit_id=request.unit_id,
            controller_type=self.controller_type,
            model_id="fake-hanwam:model",
            action={"freq_target": 40.0, "eev": 180.0, "fan_out": 750.0},
            valid_for_seconds=5,
            debug={"reused_cached_action": reused},
        )

    def reset(self) -> ResetResponse:
        count = len(self.sessions)
        self.sessions.clear()
        return ResetResponse(controller_type=self.controller_type, sessions_cleared=count)


class ApiServicerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_api_service_config("servicer/config/api_service_hanwam.yml")
        app = create_app(config, _FakeHanWAMController())
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
        self.assertEqual(payload["service"]["port"], 8080)

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
        payload = _minimal_payload()
        payload.pop("unit_id")
        response = self.client.post("/v1/plan", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "VALIDATION_ERROR")

    def test_blank_unit_id_returns_validation_error(self) -> None:
        payload = _minimal_payload()
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
