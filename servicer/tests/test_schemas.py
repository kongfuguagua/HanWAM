from __future__ import annotations

import unittest

try:
    from pydantic import ValidationError
    from servicer.api_servicer.schemas import PlanRequest
except ModuleNotFoundError as exc:  # pragma: no cover - optional service deps
    if exc.name in {"fastapi", "pydantic"}:
        raise unittest.SkipTest(f"service dependency missing: {exc.name}")
    raise


def _payload() -> dict:
    return {
        "request_id": "schema-test",
        "unit_id": "ac-test",
        "controller_type": "hanwam_wm_mpc",
        "mode": 1,
        "step_seconds": 5,
        "target_temperature_c": 27.0,
        "elapsed_seconds": 300.0,
        "deadline_seconds": 2400.0,
        "obs_history": [{"T_in": 28.0}],
        "act_history": [{"freq_target": 40.0}],
        "return_debug": True,
    }


class PlanRequestSchemaTest(unittest.TestCase):
    def test_accepts_generic_history_maps(self) -> None:
        request = PlanRequest(**_payload())
        self.assertEqual(request.controller_type, "hanwam_wm_mpc")
        self.assertEqual(request.obs_history[0]["T_in"], 28.0)

    def test_rejects_nan(self) -> None:
        payload = _payload()
        payload["obs_history"][0]["T_in"] = float("nan")
        with self.assertRaises(ValidationError):
            PlanRequest(**payload)

    def test_rejects_unknown_top_level_field(self) -> None:
        payload = _payload()
        payload["unexpected"] = 1
        with self.assertRaises(ValidationError):
            PlanRequest(**payload)


if __name__ == "__main__":
    unittest.main()
