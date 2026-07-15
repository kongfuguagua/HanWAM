"""Resolved MiniController run configuration."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "control" / "HanWAM" / "config" / "hanwam.yml"
VALID_METHODS = {"historical", "fixed", "pid", "hanwam"}
VALID_STAGES = {"env_check", "test", "train", "eval"}
VALID_MODES = {1, 3}
CONTROL_STEP_SECONDS = 5
EXPERIMENT_HORIZON_SECONDS = 4 * 60 * 60

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "hanwam_pid_compare_mode1",
        "seed": 2026,
        "output_root": "control/outputs",
        "mode": 1,
        "stages": ["train", "eval"],
        "tracking": {
            "swanlab": {
                "enabled": False,
                "project": "haier-jk-control",
                "experiment": None,
            }
        },
    },
    "data": {
        "root": "data/dataset",
        "manifest": "data/split_manifest.csv",
        "format": "grouped_status_csv",
        "encoding": "gbk",
        "group_dirs": {
            "include": ["X1", "X2", "X3", "X4", "X5", "X8", "X11", "X16", "X17"],
            "exclude": ["unclassified"],
        },
        "split": {
            "strategy": "manifest_source_round",
            "train": {"source_contains": ["1轮", "0414~0424"]},
            "val": {"source_contains": ["2轮"]},
            "test": {"source_contains": ["3轮"]},
        },
        "sampling": {
            "step_seconds": CONTROL_STEP_SECONDS,
            "interpolation_limit": 2,
            "min_run_steps": 3,
        },
        "columns": {
            "timestamp": "ts",
            "observation": [
                "freq",
                "fan_out",
                "fan_in",
                "eev",
                "T_out_coil",
                "T_in_coil",
                "T_out_discharge",
                "T_in",
                "T_out",
                "energy_cum",
                "T_set",
                "mode",
            ],
            "target_action": ["freq_target", "eev", "fan_out"],
            "actual_action": ["freq", "eev", "fan_out"],
            "goal": ["T_set", "mode"],
        },
    },
    "simulator": {
        "air_conditioner": {
            "step_seconds": CONTROL_STEP_SECONDS,
            "freq_cap_by_mode": {"1": 80.0, "3": 110.0},
            "freq_params": {},
        },
        "environment": {
            "step_seconds": CONTROL_STEP_SECONDS,
            "max_airflow_m3h": 650.0,
            "rated_fan_rpm": 1050.0,
            "air_density_kg_m3": 1.2,
            "room_heat_capacity_kj_per_c": 90.0,
            "passive_heat_tau_seconds": 14_400.0,
        },
    },
    "train": {
        "mode": 1,
        "device": "auto",
        "epochs": 40,
        "batch_size": 4096,
        "frames_per_block": 12,
        "history_blocks": 3,
        "future_blocks": 5,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "limit_transitions": 0,
        "loss": {
            "latent_weight": 1.0,
            "prober_weight": 1.0,
            "vicreg_weight": 0.05,
            "regularizer": "sigreg",
            "sigreg_num_projections": 64,
            "sigreg_mean_weight": 1.0,
            "horizon_gamma": 0.98,
            "variance_target": 1.0,
        },
    },
    "method": {
        "name": "hanwam",
        "checkpoint_template": "control/HanWAM/checkpoints/hanwam_mode{mode}.pt",
        "model": {"class_name": "HanWAMBlockControllerModel", "latent_dim": 64, "hidden_dim": 128},
        "planner": {
            "objective": "phase_energy_clamp",
            "timing": {
                "step_seconds": CONTROL_STEP_SECONDS,
                "frames_per_block": 12,
                "history_blocks": 4,
                "future_blocks": 8,
                "chunk_steps": 12,
                "control_interval_steps": 2,
                "horizon_steps": 96,
            },
            "sampling": {
                "seed": 123,
                "num_samples": 512,
                "num_iterations": 3,
                "temperature": 0.7,
                "sample_std_fraction": 0.45,
                "min_std_fraction": 0.02,
                "proposal_action_anchors": [
                    [10.0, 100.0, 750.0],
                    [15.0, 100.0, 750.0],
                    [20.0, 100.0, 750.0],
                    [25.0, 140.0, 750.0],
                    [30.0, 180.0, 750.0],
                    [35.0, 220.0, 750.0],
                    [40.0, 240.0, 750.0],
                    [45.0, 240.0, 750.0],
                    [50.0, 240.0, 750.0],
                    [55.0, 240.0, 750.0],
                    [60.0, 240.0, 750.0],
                    [65.0, 240.0, 750.0],
                    [70.0, 240.0, 750.0],
                    [75.0, 240.0, 750.0],
                    [80.0, 240.0, 750.0],
                ],
            },
            "actuator": {
                "compressor_on_threshold_hz": 10.0,
                "snap_deadband_freq": True,
            },
            "first_principles": {
                "energy": {"reference_kwh_per_hour": 0.6},
                "phase": {
                    "reach_deadline_fraction": 0.65,
                    "reach_band_c": 0.45,
                    "hold_activation_c": 0.90,
                    "clamp_upper_c": 0.35,
                    "clamp_lower_c": 0.35,
                    "clamp_center_weight": 2.0,
                    "clamp_center_scale_c": 0.45,
                    "clamp_terminal_weight": 0.15,
                    "clamp_terminal_tail_seconds": 120,
                    "clamp_terminal_projection_seconds": 600,
                },
                "action": {
                    "reference": "last_issued_command",
                    "loss": "huber",
                    "huber_delta": 0.20,
                    "pre_first_weight": 0.5,
                    "pre_sequence_weight": 0.8,
                    "hold_first_weight": 2.0,
                    "hold_sequence_weight": 1.5,
                    "post_first_weight": 2.5,
                    "post_sequence_weight": 2.0,
                    "correction_weight": 16.0,
                    "correction_base_norm": 0.04,
                    "correction_gain_norm_per_c": 0.18,
                    "correction_deadband_c": 0.12,
                    "correction_max_norm": 0.65,
                    "hold_effort_weight": 2.0,
                    "post_effort_weight": 3.0,
                    "effort_freq_weight": 1.0,
                    "effort_eev_weight": 0.25,
                    "effort_fan_weight": 0.0,
                    "hold_anchor_weight": 50.0,
                    "post_anchor_weight": 80.0,
                    "anchor_freq": 10.0,
                    "anchor_eev": 100.0,
                    "anchor_fan": 750.0,
                    "anchor_freq_deadband": 5.0,
                    "anchor_eev_deadband": 20.0,
                    "anchor_fan_deadband": 0.0,
                    "anchor_freq_weight": 1.0,
                    "anchor_eev_weight": 0.05,
                    "anchor_fan_weight": 0.0,
                    "anchor_gate_c": 0.6,
                },
                "weights": {
                    "reach": 80.0,
                    "clamp": 140.0,
                    "energy_pre": 0.9,
                    "energy_hold": 1.5,
                    "energy_post": 2.0,
                    "action_pre": 4.0,
                    "action_hold": 48.0,
                    "action_post": 64.0,
                },
                "slew": {
                    "pre_deadline": {"freq": 30, "eev": 20, "fan_out": 100},
                    "post_deadline": {"freq": 6, "eev": 10, "fan_out": 50},
                },
            },
        },
        "action_space_by_mode": {
            "1": {"freq_target": [10.0, 80.0], "eev": [100.0, 270.0], "fan_out": [750.0, 750.0]},
            "3": {"freq_target": [0.0, 110.0], "eev": [100.0, 480.0], "fan_out": [0.0, 800.0]},
        },
        "base_action_by_mode": {
            "1": {"freq_target": 40.2, "eev": 165.0, "fan_out": 750.0},
            "3": {"freq_target": 22.0, "eev": 105.0, "fan_out": 800.0},
        },
    },
    "eval": {
        "mode": 1,
        "split": "test",
        "max_runs": 4,
        "horizon_seconds": EXPERIMENT_HORIZON_SECONDS,
        "ddl_seconds": EXPERIMENT_HORIZON_SECONDS,
        "target_temperature": None,
        "comfort_band_c": 0.5,
        "scenarios": [
            {
                "name": "four_hour_experiment",
                "split": "test",
                "max_runs": 4,
                "horizon_seconds": EXPERIMENT_HORIZON_SECONDS,
            },
        ],
    },
    "visualization": {
        "enabled": True,
        "per_run_debug": True,
        "trajectory_compare": True,
        "save_csv": True,
        "dpi": 160,
    },
}


def deep_merge(defaults: dict, overrides: dict | None) -> dict:
    merged = deepcopy(defaults)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def mode_key(mode: int | str) -> str:
    if str(mode) == "all":
        return "all"
    return str(int(float(mode)))


def expand_modes(mode: int | str) -> list[int]:
    if str(mode) == "all":
        return [1, 3]
    value = int(float(mode))
    if value not in VALID_MODES:
        raise ValueError(f"mode must be 1, 3, or all; got {mode!r}")
    return [value]


def project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def checkpoint_path(config: dict, mode: int) -> Path:
    template = config["method"].get("checkpoint_template")
    if not template:
        raise ValueError("Missing method.checkpoint_template")
    return project_path(str(template).format(mode=mode))


def output_root(config: dict) -> Path:
    return project_path(config["experiment"]["output_root"])


def action_space_for_method(config: dict, mode: int | str) -> dict[str, tuple[float, float]]:
    spaces = config["method"].get("action_space_by_mode") or {}
    payload = spaces.get(mode_key(mode))
    if payload is None:
        raise KeyError(f"Missing method.action_space_by_mode for mode={mode}")
    return {col: (float(bounds[0]), float(bounds[1])) for col, bounds in payload.items()}


def validate_config(config: dict) -> None:
    if config.get("schema") is not None:
        raise ValueError("Top-level `schema` is no longer supported; use data.columns.")
    if config.get("constraints") is not None:
        raise ValueError("Top-level `constraints` is not supported; use method.action_space_by_mode.")
    if config.get("methods") is not None:
        raise ValueError("Top-level `methods` is not supported; one config describes one `method`.")
    if config["data"]["format"] != "grouped_status_csv":
        raise ValueError("Only data.format=grouped_status_csv is currently supported.")
    for mode in expand_modes(config["experiment"]["mode"]):
        pass
    for mode in expand_modes(config["train"]["mode"]):
        pass
    for mode in expand_modes(config["eval"]["mode"]):
        pass
    method = config["method"].get("name")
    if method not in VALID_METHODS:
        raise ValueError(f"method.name must be one of {sorted(VALID_METHODS)}; got {method!r}")
    if "train" in config["experiment"].get("stages", []) and method != "hanwam":
        raise ValueError("Only method.name=hanwam supports train stage.")
    columns = config["data"]["columns"]
    for key in ("observation", "target_action", "actual_action", "goal"):
        if not columns.get(key):
            raise ValueError(f"data.columns.{key} must not be empty")
    step_seconds = int(config["data"]["sampling"]["step_seconds"])
    if step_seconds != CONTROL_STEP_SECONDS:
        raise ValueError(f"data.sampling.step_seconds must be {CONTROL_STEP_SECONDS} for real control experiments.")
    if int(config["simulator"]["air_conditioner"]["step_seconds"]) != step_seconds:
        raise ValueError("simulator.air_conditioner.step_seconds must match data.sampling.step_seconds.")
    environment_step = config["simulator"].get("environment", {}).get("step_seconds", step_seconds)
    if int(environment_step) != step_seconds:
        raise ValueError("simulator.environment.step_seconds must match data.sampling.step_seconds.")
    for scenario in config["eval"].get("scenarios", []):
        horizon_seconds = int(scenario.get("horizon_seconds", config["eval"]["horizon_seconds"]))
        if horizon_seconds % step_seconds:
            raise ValueError("eval scenario horizon_seconds must be divisible by data.sampling.step_seconds.")
    if method in {"pid", "hanwam"}:
        for mode in (1, 3):
            action_space_for_method(config, mode)


def load_config(path: str | Path = CONFIG_PATH, overrides: dict | None = None) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    config = deep_merge(DEFAULT_CONFIG, loaded)
    config = deep_merge(config, overrides)
    validate_config(config)
    return config


def stage_list(config: dict, override: str | None = None) -> list[str]:
    raw = override.split(",") if override else config["experiment"].get("stages", [])
    stages = [stage.strip() for stage in raw if str(stage).strip()]
    unknown = [stage for stage in stages if stage not in VALID_STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")
    return stages
