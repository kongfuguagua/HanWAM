"""Data loading utilities for HanWAM world-model training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from control.MiniController.config_schema import project_path
from data.io import DATA_DIR, read_raw_csv, safe_source_name
from .type import WAM_ACTION_COLS, WAM_OBS_COLS, WAM_PHYSICAL_COLS, WAM_REQUIRED_RAW_COLS

PHYSICAL_COLS = WAM_PHYSICAL_COLS


@dataclass
class Run:
    name: str
    path: Path
    split: str
    frame: pd.DataFrame


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def encode(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def decode(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "Normalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def _data_config(config: dict | None) -> dict:
    if config is None:
        return {
            "root": "data/dataset",
            "manifest": "data/split_manifest.csv",
            "group_dirs": {"include": [], "exclude": []},
            "split": {
                "strategy": "manifest_source_round",
                "train": {"source_contains": ["1轮", "0414~0424"]},
                "val": {"source_contains": ["2轮"]},
                "test": {"source_contains": ["3轮"]},
            },
            "sampling": {"step_seconds": 5, "interpolation_limit": 2, "min_run_steps": 3},
        }
    return config["data"]


def columns_from_config(config: dict | None) -> dict:
    return {
        "observation": list(WAM_OBS_COLS),
        "target_action": list(WAM_ACTION_COLS),
        "actual_action": ["freq", "eev", "fan_out"],
        "goal": ["T_set", "mode"],
        "physical": list(WAM_PHYSICAL_COLS),
    }


def _manifest_map(config: dict | None) -> dict[str, str]:
    data_cfg = _data_config(config)
    manifest = data_cfg.get("manifest")
    if not manifest:
        return {}
    path = project_path(manifest)
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {str(row["destination"]): str(row["source"]) for _, row in frame.iterrows()}


def _split_for_path(path: Path, root: Path, config: dict | None, manifest: dict[str, str]) -> str:
    data_cfg = _data_config(config)
    rel = str(path.relative_to(root))
    source = manifest.get(rel, rel)
    split_cfg = data_cfg.get("split") or {}
    for split in ("train", "val", "test"):
        patterns = ((split_cfg.get(split) or {}).get("source_contains") or [])
        if any(pattern in source for pattern in patterns):
            return split
    if "3轮" in source:
        return "test"
    if "2轮" in source:
        return "val"
    return "train"


def discover_grouped_csvs(config: dict | None = None) -> list[Path]:
    data_cfg = _data_config(config)
    root = project_path(data_cfg.get("root")) or DATA_DIR
    include = set((data_cfg.get("group_dirs") or {}).get("include") or [])
    exclude = set((data_cfg.get("group_dirs") or {}).get("exclude") or [])
    paths: list[Path] = []
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        group = group_dir.name
        if include and group not in include:
            continue
        if group in exclude:
            continue
        paths.extend(sorted(group_dir.glob("status_data_*.csv")))
    return paths


def load_run(path: Path, step_seconds: int = 5, mode: int | None = 1, config: dict | None = None, split: str | None = None) -> Run:
    data_cfg = _data_config(config)
    columns = columns_from_config(config)
    interpolation_limit = int((data_cfg.get("sampling") or {}).get("interpolation_limit", 2))
    frame = read_raw_csv(path)
    needed = list(WAM_REQUIRED_RAW_COLS)
    frame = frame.dropna(subset=needed)
    frame = frame.sort_values("ts").drop_duplicates("ts", keep="last")

    numeric = [col for col in needed if col != "ts"]
    frame = (
        frame.set_index("ts")[numeric]
        .resample(f"{step_seconds}s")
        .mean()
        .interpolate(method="time", limit=interpolation_limit, limit_direction="both")
        .dropna()
        .reset_index()
    )
    if len(frame):
        frame["elapsed_seconds"] = (frame["ts"] - frame["ts"].iloc[0]).dt.total_seconds().astype(np.float32)
    if mode is not None:
        frame = frame[np.isclose(frame["mode"], float(mode), atol=0.25)].reset_index(drop=True)
        if len(frame):
            frame["elapsed_seconds"] = np.arange(len(frame), dtype=np.float32) * float(step_seconds)
    frame["freq_target"] = frame["freq_in_tgt"]
    required = columns["observation"] + columns["target_action"] + columns["actual_action"] + columns["goal"]
    frame = frame.dropna(subset=sorted(set(required))).reset_index(drop=True)

    return Run(name=safe_source_name(path, data_dir=project_path(data_cfg.get("root")) or DATA_DIR), path=path, split=split or "train", frame=frame)


def load_all_runs(
    step_seconds: int = 5,
    mode: int | None = 1,
    cooling_only: bool | None = None,
    config: dict | None = None,
) -> list[Run]:
    if cooling_only is not None:
        mode = 1 if cooling_only else None
    data_cfg = _data_config(config)
    step_seconds = int((data_cfg.get("sampling") or {}).get("step_seconds", step_seconds))
    min_run_steps = int((data_cfg.get("sampling") or {}).get("min_run_steps", 3))
    root = project_path(data_cfg.get("root")) or DATA_DIR
    manifest = _manifest_map(config)
    runs: list[Run] = []
    for path in discover_grouped_csvs(config):
        split = _split_for_path(path, root, config, manifest)
        try:
            run = load_run(path, step_seconds=step_seconds, mode=mode, config=config, split=split)
        except ValueError:
            continue
        if len(run.frame) >= min_run_steps:
            runs.append(run)
    if not runs:
        raise ValueError(f"No usable CSV runs under {root}")
    return runs


def describe_runs(runs: list[Run], step_seconds: int = 5) -> pd.DataFrame:
    rows = []
    for run in runs:
        frame = run.frame
        rows.append(
            {
                "run": run.name,
                "split": run.split,
                "rows": len(frame),
                "duration_h": max(0, len(frame) - 1) * step_seconds / 3600,
                "T_in_start": frame["T_in"].iloc[0],
                "T_in_end": frame["T_in"].iloc[-1],
                "freq_max": frame["freq"].max(),
                "freq_target_max": frame["freq_in_tgt"].max(),
                "mode": frame["mode"].mode().iloc[0] if len(frame) else None,
                "T_set": frame["T_set"].median() if len(frame) else None,
            }
        )
    return pd.DataFrame(rows)


def transition_arrays(
    runs: list[Run],
    split: str,
    obs_cols: list[str] | None = None,
    action_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    obs_cols = obs_cols or WAM_OBS_COLS
    action_cols = action_cols or WAM_ACTION_COLS
    obs, actions, next_obs, keys = [], [], [], []
    for run in runs:
        if run.split != split or len(run.frame) < 2:
            continue
        frame = run.frame
        o = frame[obs_cols].to_numpy(np.float32)
        a = frame[action_cols].to_numpy(np.float32)
        obs.append(o[:-1])
        actions.append(a[:-1])
        next_obs.append(o[1:])
        keys.extend([run.name] * (len(frame) - 1))
    if not obs:
        raise ValueError(f"No transition data for split={split}")
    return np.concatenate(obs), np.concatenate(actions), np.concatenate(next_obs), keys


def policy_transition_arrays(
    runs: list[Run],
    split: str,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns = columns_from_config(config)
    obs_cols = columns["observation"]
    actual_cols = columns["actual_action"]
    target_cols = columns["target_action"]
    goal_cols = columns["goal"]
    obs, actual_actions, target_actions, next_obs, goals, keys = [], [], [], [], [], []
    for run in runs:
        if run.split != split or len(run.frame) < 2:
            continue
        frame = run.frame
        o = frame[obs_cols].to_numpy(np.float32)
        obs.append(o[:-1])
        actual_actions.append(frame[actual_cols].to_numpy(np.float32)[:-1])
        target_actions.append(frame[target_cols].to_numpy(np.float32)[:-1])
        next_obs.append(o[1:])
        goals.append(frame[goal_cols].to_numpy(np.float32)[:-1])
        keys.extend([run.name] * (len(frame) - 1))
    if not obs:
        raise ValueError(f"No transition data for split={split}")
    return (
        np.concatenate(obs),
        np.concatenate(actual_actions),
        np.concatenate(target_actions),
        np.concatenate(next_obs),
        np.concatenate(goals),
        keys,
    )


def sequence_arrays(
    runs: list[Run],
    split: str,
    horizon_steps: int,
    config: dict | None = None,
    limit: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns = columns_from_config(config)
    obs_cols = columns["observation"]
    target_cols = columns["target_action"]
    train_cfg = (config or {}).get("train") or {}
    history_steps = int(train_cfg.get("history_steps", 1))
    obs_history, actions, future_obs_history, physical, keys = [], [], [], [], []
    horizon_steps = int(horizon_steps)
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")
    if history_steps < 1:
        raise ValueError("history_steps must be >= 1")

    eligible: list[tuple[Run, int]] = []
    for run in runs:
        if run.split != split or len(run.frame) < horizon_steps + 1:
            continue
        for start in range(history_steps - 1, len(run.frame) - horizon_steps):
            eligible.append((run, start))
    if not eligible:
        raise ValueError(f"No sequence data for split={split} horizon_steps={horizon_steps} history_steps={history_steps}")
    if limit and int(limit) > 0 and len(eligible) > int(limit):
        rng = np.random.default_rng(seed)
        selected_idx = rng.choice(len(eligible), size=int(limit), replace=False)
        eligible = [eligible[int(idx)] for idx in np.sort(selected_idx)]

    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for run, start in eligible:
        if run.name not in prepared:
            frame = run.frame.reset_index(drop=True).copy()
            frame["T_in_delta"] = frame["T_in"].diff().fillna(0.0)
            if "energy_cum" in frame:
                frame["electric_kwh_delta"] = frame["energy_cum"].diff().fillna(0.0).clip(lower=0.0)
            else:
                frame["electric_kwh_delta"] = 0.0
            prepared[run.name] = (
                frame[obs_cols].to_numpy(np.float32),
                frame[target_cols].to_numpy(np.float32),
                frame[PHYSICAL_COLS].to_numpy(np.float32),
            )
        o, a, p = prepared[run.name]
        end = start + horizon_steps
        obs_history.append(o[start - history_steps + 1 : start + 1])
        actions.append(a[start:end])
        future_obs_history.append(
            np.stack(
                [o[future - history_steps + 1 : future + 1] for future in range(start + 1, end + 1)],
                axis=0,
            )
        )
        physical.append(p[start + 1 : end + 1])
        keys.append(run.name)
    return (
        np.asarray(obs_history, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(future_obs_history, dtype=np.float32),
        np.asarray(physical, dtype=np.float32),
        keys,
    )


def start_windows(
    runs: list[Run],
    split: str,
    horizon_steps: int = 720,
) -> list[Run]:
    selected = []
    for run in runs:
        if run.split == split and len(run.frame) >= horizon_steps + 1:
            selected.append(run)
    return selected


def fit_normalizer(values: np.ndarray, eps: float = 1e-6) -> Normalizer:
    return Normalizer(
        mean=values.mean(axis=0).astype(np.float32),
        std=np.maximum(values.std(axis=0), eps).astype(np.float32),
    )


def action_bounds(
    runs: list[Run],
    quantile: tuple[float, float] = (0.01, 0.99),
    config: dict | None = None,
) -> dict[str, tuple[float, float]]:
    target_cols = columns_from_config(config)["target_action"]
    if config is not None:
        spaces = (config.get("method") or {}).get("action_space_by_mode") or {}
        mode = str((config.get("train") or {}).get("mode", (config.get("eval") or {}).get("mode", 1)))
        if mode != "all" and str(int(float(mode))) in spaces:
            payload = spaces[str(int(float(mode)))]
            return {col: (float(payload[col][0]), float(payload[col][1])) for col in target_cols}
    frames = [r.frame for r in runs if r.split in {"train", "val"}]
    data = pd.concat(frames, ignore_index=True)
    bounds = {}
    for col in target_cols:
        lo, hi = data[col].quantile(list(quantile)).to_numpy(dtype=float)
        bounds[col] = (float(lo), float(hi))
    return bounds
