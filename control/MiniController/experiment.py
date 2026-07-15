"""Shared closed-loop experiment runner for controller methods."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from control.HanWAM.dataloader import load_all_runs, start_windows
from simu.air_conditioner import AirConditionerSimulator
from simu.environment import EnvironmentSimulator

from .base import ControllerContext
from .config import action_bounds_for_mode, signed_temperature_error
from .config_schema import checkpoint_path, expand_modes, load_config, output_root
from .factory import build_controller
from .metrics import summarize_closed_loop
from .tracking import SwanLabTracker
from .visualization import plot_run_debug, plot_summary


def _saturation_vapor_pressure_kpa(temp_c: float) -> float:
    temp = float(temp_c)
    return float(0.61094 * np.exp((17.625 * temp) / (temp + 243.04)))


def relative_humidity_from_wet_bulb(
    dry_bulb_c: float,
    wet_bulb_c: float,
    pressure_kpa: float = 101.325,
) -> float:
    """Approximate RH from dry/wet bulb temperatures using a ventilated psychrometer equation."""
    dry = float(dry_bulb_c)
    wet = min(float(wet_bulb_c), dry)
    gamma = 0.00066 * (1.0 + 0.00115 * wet) * float(pressure_kpa)
    vapor_pressure = _saturation_vapor_pressure_kpa(wet) - gamma * (dry - wet)
    rh = vapor_pressure / max(_saturation_vapor_pressure_kpa(dry), 1e-9)
    return float(np.clip(rh, 0.01, 1.0))


def _condition_value(condition: dict, *names: str, default: float | None = None) -> float:
    for name in names:
        if name in condition and condition[name] is not None:
            return float(condition[name])
    if default is None:
        raise KeyError(f"Missing condition field; expected one of {names}")
    return float(default)


def standard_condition_frame(
    condition: dict,
    *,
    mode: int,
    horizon_seconds: int,
    step_seconds: int,
    name: str,
) -> pd.DataFrame:
    indoor_dry = _condition_value(
        condition,
        "indoor_initial_dry_bulb_c",
        "indoor_initial_dry_c",
        "initial_T_in_c",
    )
    indoor_wet = _condition_value(
        condition,
        "indoor_initial_wet_bulb_c",
        "indoor_initial_wet_c",
        "initial_T_in_wet_bulb_c",
    )
    target_dry = _condition_value(
        condition,
        "indoor_target_dry_bulb_c",
        "indoor_target_dry_c",
        "target_T_in_c",
    )
    outdoor_dry = _condition_value(
        condition,
        "outdoor_dry_bulb_c",
        "outdoor_dry_c",
        "T_out_c",
    )
    outdoor_wet = _condition_value(
        condition,
        "outdoor_wet_bulb_c",
        "outdoor_wet_c",
        "T_out_wet_bulb_c",
    )
    rh_in = float(condition.get("RH_in", relative_humidity_from_wet_bulb(indoor_dry, indoor_wet)))
    outdoor_rh = float(condition.get("RH_out", relative_humidity_from_wet_bulb(outdoor_dry, outdoor_wet)))
    steps = int(horizon_seconds // step_seconds) + 1
    elapsed = np.arange(steps, dtype=float) * float(step_seconds)
    timestamp = pd.Timestamp("2026-07-06 00:00:00") + pd.to_timedelta(elapsed, unit="s")
    payload = {
        "ts": timestamp,
        "I_comp": np.zeros(steps, dtype=float),
        "RH_in": np.full(steps, rh_in, dtype=float),
        "RH_out": np.full(steps, outdoor_rh, dtype=float),
        "T_in": np.full(steps, indoor_dry, dtype=float),
        "T_in_coil": np.full(steps, float(condition.get("T_in_coil", indoor_dry)), dtype=float),
        "T_out": np.full(steps, outdoor_dry, dtype=float),
        "T_out_coil": np.full(steps, float(condition.get("T_out_coil", outdoor_dry)), dtype=float),
        "T_out_discharge": np.full(steps, float(condition.get("T_out_discharge", outdoor_dry)), dtype=float),
        "T_set": np.full(steps, target_dry, dtype=float),
        "eev": np.full(steps, float(condition.get("eev", 0.0)), dtype=float),
        "energy_cum": np.zeros(steps, dtype=float),
        "fan_in": np.full(steps, float(condition.get("fan_in", 900.0)), dtype=float),
        "fan_out": np.full(steps, float(condition.get("fan_out", 0.0)), dtype=float),
        "freq": np.full(steps, float(condition.get("freq", 0.0)), dtype=float),
        "freq_in_tgt": np.full(steps, float(condition.get("freq_target", 0.0)), dtype=float),
        "freq_target": np.full(steps, float(condition.get("freq_target", 0.0)), dtype=float),
        "mode": np.full(steps, float(mode), dtype=float),
        "elapsed_seconds": elapsed,
        "condition": np.full(steps, str(condition.get("name", name)), dtype=object),
    }
    return pd.DataFrame(payload)


def observation_from_state(
    env_state: dict,
    ac_state: dict,
    target: float,
    mode: int,
    energy_cum: float,
) -> dict:
    return {
        "T_out": env_state["T_out"],
        "T_out_coil": env_state["T_out_coil"],
        "T_out_discharge": env_state.get("T_out_discharge", env_state["T_out"]),
        "T_in": env_state["T_in"],
        "T_in_coil": env_state["T_in_coil"],
        "freq_target": ac_state["freq_target"],
        "freq": ac_state["freq"],
        "eev": ac_state["eev"],
        "fan_out": ac_state["fan_out"],
        "fan_in": env_state["fan_in"],
        "RH_in": env_state["RH_in"],
        "T_set": float(target),
        "mode": float(mode),
        "energy_cum": float(energy_cum),
        "elapsed_seconds": float(env_state.get("elapsed_seconds", 0.0)),
    }


def initial_environment_state(frame: pd.DataFrame) -> dict:
    row = frame.iloc[0]
    state = row.to_dict()
    state.update(
        {
            "indoor_fan_target": float(row.get("fan_in", 900.0)),
            "RH_target": float(row.get("RH_in", 0.6)),
            "inference_freq": float(row.get("freq_target", row.get("freq", 0.0))),
            "inference_eev": float(row.get("eev", 0.0)),
            "inference_fan_out": float(row.get("fan_out", 0.0)),
            "inference_fan_in": float(row.get("fan_in", 900.0)),
        }
    )
    state.update({
        "T_out": float(row["T_out"]),
        "T_in": float(row["T_in"]),
        "T_out_coil": float(row["T_out_coil"]),
        "T_out_discharge": float(row.get("T_out_discharge", row["T_out"])),
        "T_in_coil": float(row["T_in_coil"]),
        "RH_in": float(row.get("RH_in", 0.6)),
        "fan_in": float(row.get("fan_in", 900.0)),
    })
    return state


def run_closed_loop(
    frame: pd.DataFrame,
    policy: str,
    mode: int,
    target: float,
    controller: Any,
    horizon_seconds: int,
    step_seconds: int = 5,
    simulator_config: dict | None = None,
    control_deadline_seconds: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    simulator_config = simulator_config or {}
    ac_cfg = simulator_config.get("air_conditioner") or {}
    env_cfg = simulator_config.get("environment") or {}
    freq_cap_by_mode = ac_cfg.get("freq_cap_by_mode") or {}
    freq_cap = freq_cap_by_mode.get(str(mode), ac_cfg.get("freq_cap", 90.0))
    horizon_steps = horizon_seconds // step_seconds
    ac = AirConditionerSimulator(
        mode=mode,
        step_seconds=step_seconds,
        freq_cap=freq_cap,
        energy_model=ac_cfg.get("energy_model"),
        freq_params=ac_cfg.get("freq_params") or {},
    )
    env = EnvironmentSimulator(
        mode=mode,
        step_seconds=step_seconds,
        max_airflow_m3h=float(env_cfg.get("max_airflow_m3h", 650.0)),
        rated_fan_rpm=float(env_cfg.get("rated_fan_rpm", 1050.0)),
        air_density_kg_m3=float(env_cfg.get("air_density_kg_m3", 1.2)),
        room_heat_capacity_kj_per_c=float(env_cfg.get("room_heat_capacity_kj_per_c", 90.0)),
        passive_heat_tau_seconds=float(env_cfg.get("passive_heat_tau_seconds", 14_400.0)),
        temperature_model=env_cfg.get("temperature_model"),
        room_hybrid_kwargs=env_cfg.get("room_hybrid") or {},
        roomv4_kwargs=env_cfg.get("roomv4") or env_cfg.get("room_v4") or {},
        temperature_env=env_cfg.get("temperature_env"),
    )
    ac_state = ac.reset(freq0=float(frame["freq"].iloc[0]), target0=float(frame["freq_target"].iloc[0]))
    env_state = env.reset(initial_environment_state(frame))
    ac_state["eev"] = float(frame["eev"].iloc[0])
    ac_state["fan_out"] = float(frame["fan_out"].iloc[0])
    initial_energy_cum = float(frame["energy_cum"].iloc[0]) if "energy_cum" in frame else 0.0
    simulated_energy_cum = 0.0

    rows = [
        {
            **env_state,
            **ac_state,
            "target_T_in": float(target),
            "mode": mode,
            "policy": policy,
        }
    ]
    controller_rows = []
    history_rows = []

    for step in range(horizon_steps):
        obs = observation_from_state(
            env_state,
            ac_state,
            target=target,
            mode=mode,
            energy_cum=initial_energy_cum + simulated_energy_cum,
        )
        error = signed_temperature_error(mode, obs["T_in"], target)
        debug: dict[str, Any] = {}
        if policy == "historical":
            src = frame.iloc[min(step, len(frame) - 1)]
            action = np.asarray([src["freq_target"], src["eev"], src["fan_out"]], dtype=np.float32)
        else:
            if controller is None:
                raise ValueError(f"{policy} policy requires a controller")
            context = ControllerContext(
                mode=mode,
                step_seconds=step_seconds,
                horizon_seconds=step_seconds,
                elapsed_seconds=float(env_state["elapsed_seconds"]),
                metadata={
                    "ddl_seconds": float(control_deadline_seconds if control_deadline_seconds is not None else horizon_seconds),
                    "experiment_horizon_seconds": float(horizon_seconds),
                },
            )
            plan = controller.plan(obs, target, context=context)
            action = np.asarray(plan.action, dtype=np.float32)
            debug.update(plan.debug)
            if not plan.history.empty:
                history = plan.history.copy()
                history["controller_elapsed_seconds"] = env_state["elapsed_seconds"]
                history_rows.extend(history.to_dict("records"))

        ac_state = ac.step(action)
        simulated_energy_cum += float(ac_state["electric_kwh"])
        env_state = env.step(ac_state)
        controller_rows.append(
            {
                "elapsed_seconds": ac_state["elapsed_seconds"],
                "policy": policy,
                "T_in": obs["T_in"],
                "target_T_in": float(target),
                "error": error,
                "freq_target": float(action[0]),
                "freq": ac_state["freq"],
                "eev": float(action[1]),
                "fan_out": float(action[2]),
                "power_w": ac_state["power_w"],
                "thermal_power_w": env_state["thermal_power_w"],
                "electric_kwh": ac_state["electric_kwh"],
                "thermal_kwh": env_state["thermal_kwh"],
                **debug,
            }
        )
        rows.append(
            {
                **env_state,
                **ac_state,
                "target_T_in": float(target),
                "mode": mode,
                "policy": policy,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(controller_rows), pd.DataFrame(history_rows)


def log_to_swanlab(
    tracker: SwanLabTracker,
    summary: pd.DataFrame,
    trajectories: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    if not tracker.enabled:
        return
    for idx, row in summary.reset_index(drop=True).iterrows():
        prefix = f"{row['policy']}/{row['run']}"
        tracker.log(
            {
                f"{prefix}/reach_time_s": row["reach_time_s"],
                f"{prefix}/settle_time_s": row.get("settle_time_s"),
                f"{prefix}/min_abs_error_c": row.get("min_abs_error_c"),
                f"{prefix}/final_error_c": row.get("final_error_c"),
                f"{prefix}/final_T_in_c": row["final_T_in_c"],
                f"{prefix}/energy_efficiency": row["energy_efficiency"],
                f"{prefix}/electric_kwh": row["electric_kwh"],
                f"{prefix}/thermal_kwh": row["thermal_kwh"],
            },
            step=int(idx),
        )
    for _, row in trajectories.iterrows():
        prefix = f"{row['policy']}/{row['run']}"
        step = int(row["elapsed_seconds"] // 5)
        tracker.log(
            {
                f"{prefix}/T_in": row["T_in"],
                f"{prefix}/freq_target": row["freq_target"],
                f"{prefix}/freq": row["freq"],
                f"{prefix}/power_w": row["power_w"],
                f"{prefix}/thermal_power_w": row["thermal_power_w"],
            },
            step=step,
        )
    for path in plot_paths:
        tracker.log_image(f"plots/{path.stem}", path, caption=path.stem)


def _scenario_payload(eval_cfg: dict) -> list[dict]:
    scenarios = eval_cfg.get("scenarios") or []
    if scenarios:
        return scenarios
    return [
        {
            "name": "default",
            "split": eval_cfg["split"],
            "max_runs": eval_cfg["max_runs"],
            "horizon_seconds": eval_cfg["horizon_seconds"],
            "target_temperature": eval_cfg.get("target_temperature"),
        }
    ]


def run_eval_from_config(config: dict) -> dict:
    np.random.seed(int(config["experiment"]["seed"]))
    eval_cfg = config["eval"]
    visual_cfg = config["visualization"]
    step_seconds = int(config["data"]["sampling"]["step_seconds"])
    method = config["method"]["name"]
    results = {}

    tracker_cfg = config["experiment"]["tracking"]["swanlab"]
    tracker = SwanLabTracker(
        enabled=bool(tracker_cfg["enabled"]),
        project=tracker_cfg["project"],
        experiment_name=tracker_cfg.get("experiment") or config["experiment"]["name"],
        config=config,
    )

    try:
        for mode in expand_modes(eval_cfg["mode"]):
            try:
                runs = load_all_runs(mode=mode, config=config)
            except ValueError as exc:
                print(f"[warn] skip eval mode={mode}: {exc}")
                continue
            bounds = action_bounds_for_mode(config, mode)
            checkpoint = checkpoint_path(config, mode) if method == "hanwam" else None
            args = SimpleNamespace(checkpoint=checkpoint)

            for scenario in _scenario_payload(eval_cfg):
                scenario_name = scenario.get("name", "default")
                split = scenario.get("split", eval_cfg["split"])
                horizon_seconds = int(scenario.get("horizon_seconds", eval_cfg["horizon_seconds"]))
                ddl_seconds = float(scenario.get("ddl_seconds", eval_cfg.get("ddl_seconds", horizon_seconds)))
                control_deadline_seconds = scenario.get(
                    "control_deadline_seconds",
                    scenario.get("reach_deadline_seconds", eval_cfg.get("control_deadline_seconds")),
                )
                control_deadline_seconds = None if control_deadline_seconds is None else float(control_deadline_seconds)
                max_runs = int(scenario.get("max_runs", eval_cfg["max_runs"]))
                condition = scenario.get("condition")
                if condition:
                    frame = standard_condition_frame(
                        condition,
                        mode=mode,
                        horizon_seconds=horizon_seconds,
                        step_seconds=step_seconds,
                        name=scenario_name,
                    )
                    selected = [
                        SimpleNamespace(
                            name=str(condition.get("name", scenario_name)),
                            frame=frame,
                            split="standard_condition",
                            path=None,
                        )
                    ]
                else:
                    selected = start_windows(runs, split, horizon_steps=horizon_seconds // step_seconds)[:max_runs]
                if not selected:
                    raise ValueError(f"No split={split} mode={mode} runs with requested horizon")

                output_dir = output_root(config) / config["experiment"]["name"] / scenario_name / f"mode{mode}"
                output_dir.mkdir(parents=True, exist_ok=True)
                summaries = []
                trajectories = []
                controller_logs = []
                history_logs = []
                plot_paths: list[Path] = []

                if method == "hanwam" and not checkpoint.exists():
                    print(f"[warn] skip method=hanwam because checkpoint does not exist: {checkpoint}")
                    continue
                controller = build_controller(method, mode, config, bounds, args)
                for run in selected:
                    if hasattr(controller, "reset"):
                        controller.reset()
                    frame = run.frame.iloc[: horizon_seconds // step_seconds + 1].reset_index(drop=True)
                    target_override = scenario.get("target_temperature", eval_cfg.get("target_temperature"))
                    target = float(target_override if target_override is not None else frame["T_set"].iloc[0])
                    trajectory, controller_log, history_log = run_closed_loop(
                        frame=frame,
                        policy=method,
                        mode=mode,
                        target=target,
                        controller=controller,
                        horizon_seconds=horizon_seconds,
                        step_seconds=step_seconds,
                        simulator_config=config["simulator"],
                        control_deadline_seconds=control_deadline_seconds,
                    )
                    summary = summarize_closed_loop(
                        trajectory,
                        target,
                        comfort_band_c=float(eval_cfg["comfort_band_c"]),
                        comfort_lower_band_c=float(
                            scenario.get(
                                "comfort_lower_band_c",
                                eval_cfg.get("comfort_lower_band_c", eval_cfg["comfort_band_c"]),
                            )
                        ),
                        comfort_upper_band_c=float(
                            scenario.get(
                                "comfort_upper_band_c",
                                eval_cfg.get("comfort_upper_band_c", eval_cfg["comfort_band_c"]),
                            )
                        ),
                        ddl_seconds=ddl_seconds,
                        require_final_in_band=bool(
                            scenario.get(
                                "success_requires_final_band",
                                eval_cfg.get("success_requires_final_band", True),
                            )
                        ),
                        reach_deadline_seconds=scenario.get(
                            "reach_deadline_seconds",
                            eval_cfg.get("reach_deadline_seconds"),
                        ),
                        require_post_reach_band=bool(
                            scenario.get(
                                "success_requires_post_reach_band",
                                eval_cfg.get("success_requires_post_reach_band", False),
                            )
                        ),
                        require_post_deadline_band=bool(
                            scenario.get(
                                "success_requires_post_deadline_band",
                                eval_cfg.get("success_requires_post_deadline_band", False),
                            )
                        ),
                    )
                    summary.update({"run": run.name, "policy": method, "mode": mode, "target_T_in": target, "scenario": scenario_name})
                    summaries.append(summary)
                    trajectories.append(trajectory.assign(run=run.name, scenario=scenario_name))
                    controller_logs.append(controller_log.assign(run=run.name, mode=mode, scenario=scenario_name))
                    if not history_log.empty:
                        history_logs.append(history_log.assign(run=run.name, policy=method, mode=mode, scenario=scenario_name))
                    if visual_cfg.get("enabled") and visual_cfg.get("per_run_debug"):
                        plot_paths.append(
                            plot_run_debug(
                                trajectory,
                                output_dir / f"{method}_{run.name}_debug.png",
                                title=f"{method} {run.name}",
                            )
                        )

                summary_frame = pd.DataFrame(summaries)
                trajectory_frame = pd.concat(trajectories, ignore_index=True)
                controller_log_frame = pd.concat(controller_logs, ignore_index=True)
                history_log_frame = pd.concat(history_logs, ignore_index=True) if history_logs else pd.DataFrame()

                if visual_cfg.get("save_csv"):
                    summary_frame.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
                    trajectory_frame.to_csv(output_dir / "trajectories.csv", index=False, encoding="utf-8-sig")
                    controller_log_frame.to_csv(output_dir / "controller_log.csv", index=False, encoding="utf-8-sig")
                    if not history_log_frame.empty:
                        history_log_frame.to_csv(output_dir / "controller_history.csv", index=False, encoding="utf-8-sig")
                if visual_cfg.get("enabled") and visual_cfg.get("trajectory_compare"):
                    plot_paths.append(plot_summary(summary_frame, output_dir / "summary_compare.png", trajectory_frame))
                log_to_swanlab(tracker, summary_frame, trajectory_frame, plot_paths)
                results[f"{scenario_name}/mode{mode}"] = {
                    "output_dir": str(output_dir),
                    "summary": summaries,
                    "plots": [str(path) for path in plot_paths],
                }
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return results
    finally:
        tracker.finish()


if __name__ == "__main__":
    from .main import main

    main()
