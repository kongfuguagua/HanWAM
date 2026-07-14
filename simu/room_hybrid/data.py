"""Dataset utilities for V5 whole-run open-loop identification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .v5_model import CONTROL_COLUMNS, DT_SECONDS, INITIAL_COLUMNS


OUTPUT_COLUMNS = [
    "T_in",
    "T_in_coil",
    "T_out_coil",
    "T_out_discharge",
]


RAW_COLUMNS = [
    "ts", "T_out", "T_out_coil", "T_out_discharge",
    "compressor_frequency", "eev_opening", "outdoor_fan_speed",
    "I_comp", "T_in", "T_in_coil", "indoor_fan_target", "RH_target",
    "RH_in", "T_set", "mode", "energy_cum", "inference_freq",
    "inference_eev", "inference_fan_out", "inference_fan_in", "fault",
    "swing", "pid_freq", "pid_target",
]

CHINESE_COLUMN_MAP = {
    "时间戳": "ts",
    "室外温度": "T_out",
    "室外盘管温度（除霜温度）": "T_out_coil",
    "室外吐气温度": "T_out_discharge",
    "压缩机运行频率": "compressor_frequency",
    "电子膨胀阀开度": "eev_opening",
    "室外风机转速": "outdoor_fan_speed",
    "压机电流": "I_comp",
    "室内环境温度": "T_in",
    "室内盘管温度": "T_in_coil",
    "室内风机转速": "indoor_fan_target",
    "室内目标湿度": "RH_target",
    "室内实际湿度": "RH_in",
    "室内设定温度": "T_set",
    "工作模式": "mode",
    "累计电量": "energy_cum",
    "推理频率": "inference_freq",
    "推理膨胀阀开度": "inference_eev",
    "推理外风机转速": "inference_fan_out",
    "推理内风机转速": "inference_fan_in",
    "发送故障码": "fault",
    "摆风": "swing",
    "内机PID计算频率": "pid_freq",
    "内机目标频率": "pid_target",
    "下发目标频率": "command_frequency",
    "下发电子膨胀阀": "command_eev",
    "下发室外风机": "command_fan_out",
}

TARGET_LIKE_COLUMNS = {
    "T_set", "RH_target", "indoor_fan_target", "pid_freq", "pid_target",
    "inference_freq", "inference_eev", "inference_fan_out", "inference_fan_in",
}
assert not TARGET_LIKE_COLUMNS.intersection(INITIAL_COLUMNS)


def load_status_csv(path: str | Path, repair_coarse_timestamps: bool = True) -> pd.DataFrame:
    """Load both curated GBK status CSVs and newer UTF-8 full telemetry CSVs."""
    path = Path(path)
    errors: list[str] = []
    for encoding in ("gbk", "utf-8-sig"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue

        if set(CHINESE_COLUMN_MAP).intersection(frame.columns):
            frame = frame.rename(columns={
                source: target for source, target in CHINESE_COLUMN_MAP.items()
                if source in frame.columns
            })
        elif len(frame.columns) >= len(RAW_COLUMNS):
            frame = frame.iloc[:, :len(RAW_COLUMNS)].copy()
            frame.columns = RAW_COLUMNS

        if not set(RAW_COLUMNS).issubset(frame.columns):
            errors.append(f"{encoding}: missing normalized columns")
            continue

        frame = frame.copy()
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        for column in [name for name in frame.columns if name != "ts"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        def control_source(command: str, actual: str, valid_min: float, valid_max: float) -> pd.Series:
            # Curated 24-column dataset files contain inference/controller
            # intent fields, not guaranteed executed commands.  Use those only
            # when the full telemetry export has explicit command columns.
            if command in frame.columns:
                result = frame[command].copy()
                result = result.where(result.between(valid_min, valid_max), np.nan)
                result = result.where(np.isfinite(result), frame[actual])
                return result
            return frame[actual]

        frame["control_frequency"] = control_source(
            "command_frequency", "compressor_frequency", 0.0, 150.0,
        )
        frame["control_eev"] = control_source(
            "command_eev", "eev_opening", 0.0, 600.0,
        )
        frame["control_fan_out"] = control_source(
            "command_fan_out", "outdoor_fan_speed", 0.0, 1200.0,
        )
        frame = frame[[*RAW_COLUMNS, *CONTROL_COLUMNS]].copy()
        frame = frame.dropna(subset=["ts", "T_in", *CONTROL_COLUMNS]).reset_index(drop=True)

        unique_ratio = frame["ts"].nunique() / max(1, len(frame))
        minute_level = bool((frame["ts"].dt.second == 0).all())
        reconstructed = bool(
            repair_coarse_timestamps and unique_ratio < 0.5 and minute_level
        )
        if reconstructed:
            frame["ts"] = frame["ts"].iloc[0] + pd.to_timedelta(
                np.arange(len(frame), dtype=float) * DT_SECONDS, unit="s",
            )
        else:
            frame = (
                frame.sort_values("ts")
                .drop_duplicates("ts", keep="last")
                .reset_index(drop=True)
            )

        frame[[*RAW_COLUMNS[1:], *CONTROL_COLUMNS]] = frame[
            [*RAW_COLUMNS[1:], *CONTROL_COLUMNS]
        ].interpolate(
            limit=2, limit_direction="both",
        )
        frame = frame.dropna(subset=["T_in", *CONTROL_COLUMNS]).reset_index(drop=True)
        frame.attrs["timestamp_reconstructed"] = reconstructed
        frame.attrs["original_timestamp_unique_ratio"] = float(unique_ratio)
        return frame

    raise ValueError(f"{path}: cannot load status CSV ({'; '.join(errors)})")


def discover_v5_cooling_runs(data_dir: str | Path) -> list[dict]:
    """Load runs that are predominantly cooling, including mode-4 startup rows."""
    root = Path(data_dir)
    runs = []
    for path in sorted(root.rglob("*.csv")):
        frame = load_status_csv(path)
        if frame.empty:
            continue
        cooling_fraction = float((frame["mode"].round() == 1).mean())
        if cooling_fraction < 0.90:
            continue
        runs.append({
            "path": path,
            "name": str(path.relative_to(root)),
            "group": path.parent.name,
            "frame": frame,
            "cooling_fraction": cooling_fraction,
        })
    return runs


def assign_v5_splits(runs: list[dict]) -> None:
    """Hold out the newest complete run in every repeated condition group."""
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["group"], []).append(run)
    for group_runs in groups.values():
        ordered = sorted(group_runs, key=lambda run: run["frame"]["ts"].iloc[0])
        for run in ordered:
            run["split"] = "train"
        if len(ordered) >= 2:
            ordered[-1]["split"] = "test"


def compute_normalization(runs: list[dict]) -> dict[str, np.ndarray]:
    initial = np.stack([
        run["frame"].iloc[0][INITIAL_COLUMNS].to_numpy(np.float32) for run in runs
    ])
    controls = np.concatenate([
        run["frame"][CONTROL_COLUMNS].to_numpy(np.float32) for run in runs
    ])
    initial_mean = np.nanmedian(initial, axis=0).astype(np.float32)
    initial = np.where(np.isfinite(initial), initial, initial_mean)
    initial_scale = np.nanstd(initial, axis=0).astype(np.float32)
    control_mean = np.nanmean(controls, axis=0).astype(np.float32)
    control_scale = np.nanstd(controls, axis=0).astype(np.float32)
    initial_scale = np.where(initial_scale < 1e-3, 1.0, initial_scale).astype(np.float32)
    control_scale = np.where(control_scale < 1e-3, 1.0, control_scale).astype(np.float32)
    return {
        "initial_mean": initial_mean,
        "initial_scale": initial_scale,
        "control_mean": control_mean,
        "control_scale": control_scale,
    }


def run_arrays(run: dict, initial_median: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = run["frame"]
    initial = frame.iloc[0][INITIAL_COLUMNS].to_numpy(np.float32)
    initial = np.where(np.isfinite(initial), initial, initial_median).astype(np.float32)
    controls = frame[CONTROL_COLUMNS].to_numpy(np.float32)[:-1]
    truth = frame["T_in"].to_numpy(np.float32)
    return initial, controls, truth


def describe_runs(runs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "run": run["name"], "group": run["group"], "split": run.get("split", ""),
        "samples": len(run["frame"]),
        "duration_h": (len(run["frame"]) - 1) * DT_SECONDS / 3600.0,
        "cooling_fraction": run["cooling_fraction"],
    } for run in runs])


# Backward-compatible aliases for copied plotting utilities.
discover_v4_cooling_runs = discover_v5_cooling_runs
assign_v4_splits = assign_v5_splits
discover_cooling_runs = discover_v5_cooling_runs
assign_splits = assign_v5_splits
