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
                "T_out",
                "T_out_coil",
                "T_in",
                "T_in_coil",
                "freq_target",
                "freq",
                "eev",
                "fan_out",
                "fan_in",
                "RH_in",
                "T_set",
                "mode",
                "energy_cum",
                "elapsed_seconds",
            ],
            "target_action": ["freq_target", "eev", "fan_out"],
            "actual_action": ["freq", "eev", "fan_out"],
            "goal": ["T_set", "mode"],
        },
    },
    "simulator": {
        "air_conditioner": {
            "step_seconds": CONTROL_STEP_SECONDS,
            "freq_cap_by_mode": {"1": 90.0, "3": 110.0},
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
        "horizon_steps": 60,
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
        "model": {"class_name": "HanWAM", "latent_dim": 64, "hidden_dim": 128},
        "planner": {
            "objective": "hanwam_simple",
            "horizon_steps": 24,
            "chunk_steps": 6,
            "num_samples": 128,
            "num_iterations": 3,
            "elite_ratio": 0.1,
            "step_seconds": CONTROL_STEP_SECONDS,
            "reference_schedule": "deadline_linear",
            "deadline_fraction": 0.3,
            "compressor_on_threshold_hz": 15.0,
            "snap_deadband_freq": True,
            "cost_weights": {
                "tracking": 4.0,
                "energy": 2.0,
                "action_smooth": 0.05,
            },
        },
        "action_space_by_mode": {
            "1": {"freq_target": [0.0, 90.0], "eev": [69.0, 480.0], "fan_out": [0.0, 850.0]},
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
