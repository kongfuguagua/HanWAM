from __future__ import annotations

from types import SimpleNamespace
import threading
import unittest

import numpy as np

try:
    from control.HanWAM.type import WAM_ACTION_COLS, WAM_OBS_COLS
    from control.HanWAM.utils import cooling_ddl_seconds
    from servicer.api_servicer.controllers.hanwam_wm_mpc import HanWAMWMMPCController
    from servicer.api_servicer.schemas import PlanRequest
except ModuleNotFoundError as exc:  # pragma: no cover - optional service deps
    if exc.name in {"fastapi", "pydantic", "torch", "yaml"}:
        raise unittest.SkipTest(f"service dependency missing: {exc.name}")
    raise


class _DummyPlanner:
    planner_name = "mppi"
    horizon_steps = 60
    control_interval_steps = 1

    def __init__(self) -> None:
        self._lo = np.asarray([0.0, 69.0, 0.0], dtype=np.float32)
        self._hi = np.asarray([80.0, 480.0, 900.0], dtype=np.float32)
        self.reset_count = 0
        self.remaining_seconds_seen: list[float | None] = []

    def reset(self) -> None:
        self.reset_count += 1

    def plan(self, *args, remaining_seconds: float | None = None, **kwargs):
        self.remaining_seconds_seen.append(remaining_seconds)
        action = np.asarray([40.0, 165.0, 750.0], dtype=np.float32)
        sequence = np.repeat(action.reshape(1, -1), self.horizon_steps, axis=0)
        return SimpleNamespace(
            action=action,
            best_sequence=sequence,
            debug={"hanwam_planner": "mppi", "hanwam_plan_ms": 1.0},
        )


def _obs_frame(t_in: float = 28.35, t_out: float = 35.0, t_set: float = 27.0) -> dict[str, float]:
    values = {
        "freq": 40.0,
        "fan_out": 750.0,
        "fan_in": 900.0,
        "eev": 165.0,
        "T_out_coil": 35.0,
        "T_in_coil": 25.0,
        "T_out_discharge": 55.0,
        "T_in": float(t_in),
        "T_out": float(t_out),
        "energy_cum": 0.0,
        "T_set": float(t_set),
        "mode": 1.0,
    }
    return {col: values[col] for col in WAM_OBS_COLS}


def _act_frame() -> dict[str, float]:
    values = {"freq_target": 40.0, "eev": 165.0, "fan_out": 750.0}
    return {col: values[col] for col in WAM_ACTION_COLS}


def _payload(
    *,
    target: float = 27.0,
    elapsed_seconds: float = 300.0,
    deadline_seconds: float | None = None,
    first_t_in: float = 28.35,
    first_t_out: float = 35.0,
) -> dict:
    payload = {
        "request_id": "deadline-test",
        "unit_id": "ac-test",
        "controller_type": "hanwam_wm_mpc",
        "mode": 1,
        "step_seconds": 5,
        "target_temperature_c": target,
        "elapsed_seconds": elapsed_seconds,
        "obs_history": [
            _obs_frame(t_in=first_t_in, t_out=first_t_out, t_set=target),
            _obs_frame(t_in=28.0, t_out=35.0, t_set=target),
        ],
        "act_history": [_act_frame(), _act_frame()],
        "return_debug": False,
    }
    if deadline_seconds is not None:
        payload["deadline_seconds"] = deadline_seconds
    return payload


def _controller() -> HanWAMWMMPCController:
    controller = object.__new__(HanWAMWMMPCController)
    controller.config = SimpleNamespace(logging=SimpleNamespace(payloads=False))
    controller._controller_type = "hanwam_wm_mpc"
    controller.mode = 1
    controller.step_seconds = 5
    controller.history_steps = 2
    controller.frames_per_block = 2
    controller.history_blocks = 1
    controller.rollout_steps = 60
    controller.obs_cols = list(WAM_OBS_COLS)
    controller.action_cols = list(WAM_ACTION_COLS)
    controller.model_id = "dummy-model"
    controller._sessions = {}
    controller._sessions_lock = threading.Lock()
    controller._new_planner = _DummyPlanner
    return controller


class HanWAMDeadlineTest(unittest.TestCase):
    def test_cooling_ddl_seconds_formula(self) -> None:
        self.assertAlmostEqual(cooling_ddl_seconds(28.35, 35.0, 27.0), 2035.0, places=6)

    def test_plan_uses_auto_deadline_and_records_request_deadline(self) -> None:
        controller = _controller()
        response = controller.plan(PlanRequest(**_payload(deadline_seconds=120.0)))
        session = controller._sessions["ac-test"]

        self.assertAlmostEqual(response.debug["effective_deadline_seconds"], 2035.0, places=6)
        self.assertEqual(response.debug["deadline_source"], "auto_first_frame")
        self.assertTrue(response.debug["deadline_recomputed"])
        self.assertEqual(response.debug["request_deadline_seconds"], 120.0)
        self.assertAlmostEqual(session.planner.remaining_seconds_seen[-1], 1735.0, places=6)

    def test_reuses_deadline_until_target_changes(self) -> None:
        controller = _controller()
        first = controller.plan(PlanRequest(**_payload(elapsed_seconds=0.0)))
        second = controller.plan(
            PlanRequest(**_payload(elapsed_seconds=5.0, first_t_in=31.0, first_t_out=40.0))
        )

        self.assertTrue(first.debug["deadline_recomputed"])
        self.assertFalse(second.debug["deadline_recomputed"])
        self.assertAlmostEqual(second.debug["effective_deadline_seconds"], 2035.0, places=6)

        session = controller._sessions["ac-test"]
        session.cached_actions = [np.asarray([10.0, 100.0, 100.0], dtype=np.float32)]
        third = controller.plan(
            PlanRequest(
                **_payload(
                    target=26.0,
                    elapsed_seconds=0.0,
                    first_t_in=29.0,
                    first_t_out=36.0,
                )
            )
        )

        expected = cooling_ddl_seconds(29.0, 36.0, 26.0)
        self.assertTrue(third.debug["deadline_recomputed"])
        self.assertFalse(third.debug["reused_cached_action"])
        self.assertEqual(session.planner.reset_count, 1)
        self.assertEqual(session.cached_actions, [])
        self.assertAlmostEqual(third.debug["effective_deadline_seconds"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
