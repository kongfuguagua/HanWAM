from __future__ import annotations

import json
from pathlib import Path
import unittest

try:
    from pydantic import ValidationError
    from servicer.api_servicer.config import load_api_service_config
    from servicer.api_servicer.controllers import build_controller
    from servicer.api_servicer.errors import ServiceError
    from servicer.api_servicer.schemas import PlanRequest
except ModuleNotFoundError as exc:  # pragma: no cover - optional service deps
    if exc.name in {"fastapi", "pydantic"}:
        raise unittest.SkipTest(f"service dependency missing: {exc.name}")
    raise


def _payload() -> dict:
    return json.loads(Path("servicer/stubs/sample_plan_request.json").read_text(encoding="utf-8"))


class HanWAMAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_api_service_config("servicer/config/api_service_hanwam.yml")
        if not config.controller.checkpoint.exists():
            raise unittest.SkipTest(f"external checkpoint missing: {config.controller.checkpoint}")
        cls.controller = build_controller(config)

    def setUp(self) -> None:
        self.controller.reset()

    def test_metadata_is_ready(self) -> None:
        metadata = self.controller.metadata()
        self.assertEqual(metadata["controller_type"], "hanwam_wm_mpc")
        self.assertEqual(metadata["history_steps"], 36)
        self.assertEqual(metadata["rollout_steps"], 60)
        self.assertIn("checkpoint_sha256", metadata)

    def test_plan_returns_bounded_action(self) -> None:
        response = self.controller.plan(PlanRequest(**_payload()))
        action = response.action
        self.assertGreaterEqual(action["freq_target"], 0.0)
        self.assertLessEqual(action["freq_target"], 80.0)
        self.assertGreaterEqual(action["eev"], 69.0)
        self.assertLessEqual(action["eev"], 480.0)
        self.assertGreaterEqual(action["fan_out"], 0.0)
        self.assertLessEqual(action["fan_out"], 900.0)
        self.assertIn("plan_ms", response.debug)
        self.assertFalse(response.debug["reused_cached_action"])

    def test_second_request_reuses_unit_session_cached_action(self) -> None:
        payload = _payload()
        first = self.controller.plan(PlanRequest(**payload))
        second = self.controller.plan(PlanRequest(**payload))
        self.assertFalse(first.debug["reused_cached_action"])
        self.assertTrue(second.debug["reused_cached_action"])
        self.assertEqual(second.debug["session_plan_count"], 1)
        self.assertEqual(second.debug["session_cache_hit_count"], 1)

    def test_reset_clears_sessions(self) -> None:
        self.controller.plan(PlanRequest(**_payload()))
        metadata = self.controller.metadata()
        self.assertEqual(metadata["active_sessions"], 1)
        reset = self.controller.reset()
        self.assertEqual(reset.sessions_cleared, 1)
        self.assertEqual(self.controller.metadata()["active_sessions"], 0)

    def test_short_history_returns_adapter_validation_error(self) -> None:
        payload = _payload()
        payload["obs_history"] = payload["obs_history"][:-1]
        with self.assertRaises(ServiceError) as ctx:
            self.controller.plan(PlanRequest(**payload))
        self.assertEqual(ctx.exception.error_code, "VALIDATION_ERROR")

    def test_missing_unit_id_returns_schema_validation_error(self) -> None:
        payload = _payload()
        payload.pop("unit_id")
        with self.assertRaises(ValidationError):
            PlanRequest(**payload)

    def test_blank_unit_id_returns_schema_validation_error(self) -> None:
        payload = _payload()
        payload["unit_id"] = " "
        with self.assertRaises(ValidationError):
            PlanRequest(**payload)

    def test_controller_type_mismatch_returns_explicit_error(self) -> None:
        payload = _payload()
        payload["controller_type"] = "other_controller"
        with self.assertRaises(ServiceError) as ctx:
            self.controller.plan(PlanRequest(**payload))
        self.assertEqual(ctx.exception.error_code, "UNSUPPORTED_REQUEST")


if __name__ == "__main__":
    unittest.main()
