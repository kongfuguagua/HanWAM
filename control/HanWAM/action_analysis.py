"""Analyze HanWAM closed-loop action quality for deployment-oriented review."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze controller actions, compressor cycles, and off-state cooling")
    parser.add_argument("--rollout-dir", type=Path, required=True, help="Directory containing controller_log.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --rollout-dir")
    parser.add_argument("--on-threshold-hz", type=float, default=10.0)
    parser.add_argument("--actual-on-threshold-hz", type=float, default=1.0)
    parser.add_argument("--short-cycle-seconds", type=float, default=180.0)
    parser.add_argument("--high-fan-threshold", type=float, default=500.0)
    parser.add_argument("--comfort-band-c", type=float, default=0.5)
    parser.add_argument("--high-freq-threshold-hz", type=float, default=60.0)
    parser.add_argument("--lookahead-seconds", type=float, default=600.0)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional HanWAM checkpoint for action z-score diagnostics")
    parser.add_argument("--ood-z-threshold", type=float, default=2.0)
    parser.add_argument("--extreme-z-threshold", type=float, default=3.0)
    return parser.parse_args()


def _cycle_durations(elapsed: np.ndarray, on: np.ndarray) -> tuple[list[float], list[float]]:
    if len(elapsed) == 0:
        return [], []
    if len(elapsed) > 1:
        step_seconds = float(np.median(np.diff(elapsed)))
    else:
        step_seconds = 5.0
    on_durations: list[float] = []
    off_durations: list[float] = []
    start = 0
    current = bool(on[0])
    for idx in range(1, len(on)):
        if bool(on[idx]) != current:
            duration = float(elapsed[idx - 1] - elapsed[start] + step_seconds)
            (on_durations if current else off_durations).append(duration)
            start = idx
            current = bool(on[idx])
    duration = float(elapsed[-1] - elapsed[start] + step_seconds)
    (on_durations if current else off_durations).append(duration)
    return on_durations, off_durations


def _duration_stats(prefix: str, durations: list[float]) -> dict:
    if not durations:
        return {
            f"{prefix}_cycle_count": 0,
            f"{prefix}_duration_mean_s": np.nan,
            f"{prefix}_duration_p10_s": np.nan,
            f"{prefix}_duration_min_s": np.nan,
        }
    arr = np.asarray(durations, dtype=float)
    return {
        f"{prefix}_cycle_count": int(len(arr)),
        f"{prefix}_duration_mean_s": float(arr.mean()),
        f"{prefix}_duration_p10_s": float(np.quantile(arr, 0.10)),
        f"{prefix}_duration_min_s": float(arr.min()),
    }


def analyze_actions(
    controller_log: pd.DataFrame,
    *,
    on_threshold_hz: float = 15.0,
    actual_on_threshold_hz: float = 1.0,
    short_cycle_seconds: float = 180.0,
    high_fan_threshold: float = 500.0,
) -> pd.DataFrame:
    rows = []
    for run, frame in controller_log.sort_values(["run", "elapsed_seconds"]).groupby("run", sort=False):
        elapsed = frame["elapsed_seconds"].to_numpy(dtype=float)
        hours = max(float(elapsed[-1] - elapsed[0] + np.median(np.diff(elapsed))) / 3600.0, 1e-9) if len(elapsed) > 1 else 0.0
        freq_target = frame["freq_target"].to_numpy(dtype=float)
        actual_freq = frame["freq"].to_numpy(dtype=float) if "freq" in frame else np.full(len(frame), np.nan)
        fan_out = frame["fan_out"].to_numpy(dtype=float) if "fan_out" in frame else np.full(len(frame), np.nan)
        eev = frame["eev"].to_numpy(dtype=float) if "eev" in frame else np.full(len(frame), np.nan)
        delta_freq = np.abs(np.diff(freq_target)) if len(freq_target) > 1 else np.asarray([], dtype=float)

        target_on = freq_target >= float(on_threshold_hz)
        actual_on = actual_freq > float(actual_on_threshold_hz)
        starts = int(np.logical_and(actual_on[1:], ~actual_on[:-1]).sum()) if len(actual_on) > 1 else 0
        stops = int(np.logical_and(~actual_on[1:], actual_on[:-1]).sum()) if len(actual_on) > 1 else 0
        on_durations, off_durations = _cycle_durations(elapsed, actual_on)
        short_on = [duration for duration in on_durations if duration < float(short_cycle_seconds)]

        off_mask = ~actual_on
        high_fan_off = off_mask & (fan_out > float(high_fan_threshold))
        rows.append(
            {
                "run": run,
                "steps": int(len(frame)),
                "duration_h": float(hours),
                "mean_freq_target": float(np.nanmean(freq_target)),
                "mean_actual_freq": float(np.nanmean(actual_freq)),
                "mean_abs_delta_freq_target": float(delta_freq.mean()) if len(delta_freq) else 0.0,
                "jump_gt5_ratio": float((delta_freq > 5.0).mean()) if len(delta_freq) else 0.0,
                "jump_gt10_ratio": float((delta_freq > 10.0).mean()) if len(delta_freq) else 0.0,
                "freq_target_lt_on_threshold_ratio": float((freq_target < float(on_threshold_hz)).mean()),
                "actual_freq_off_ratio": float(off_mask.mean()),
                "target_on_ratio": float(target_on.mean()),
                "actual_on_ratio": float(actual_on.mean()),
                "start_count": starts,
                "stop_count": stops,
                "starts_per_hour": float(starts / hours) if hours > 0.0 else np.nan,
                "short_on_cycle_count": int(len(short_on)),
                "short_on_cycle_ratio": float(len(short_on) / max(len(on_durations), 1)),
                "off_high_fan_ratio": float(high_fan_off.mean()),
                "mean_fan_out": float(np.nanmean(fan_out)),
                "mean_eev": float(np.nanmean(eev)),
                **_duration_stats("on", on_durations),
                **_duration_stats("off", off_durations),
            }
        )
    return pd.DataFrame(rows)


def analyze_off_cooling(
    trajectories: pd.DataFrame,
    *,
    actual_on_threshold_hz: float = 1.0,
    power_threshold_w: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for run, frame in trajectories.sort_values(["run", "elapsed_seconds"]).groupby("run", sort=False):
        temp = frame["T_in"].to_numpy(dtype=float)
        if len(temp) < 2:
            continue
        next_delta = np.diff(temp)
        freq = frame["freq"].to_numpy(dtype=float)[:-1]
        power = frame["power_w"].to_numpy(dtype=float)[:-1] if "power_w" in frame else np.zeros(len(next_delta))
        mask = (freq <= float(actual_on_threshold_hz)) & (power <= float(power_threshold_w))
        cooling = next_delta[mask & (next_delta < 0.0)]
        rows.append(
            {
                "run": run,
                "off_power_zero_steps": int(mask.sum()),
                "off_power_zero_cooling_steps": int(len(cooling)),
                "off_power_zero_cooling_ratio": float(len(cooling) / max(int(mask.sum()), 1)),
                "off_power_zero_mean_delta_c": float(next_delta[mask].mean()) if mask.any() else np.nan,
                "off_power_zero_min_delta_c": float(next_delta[mask].min()) if mask.any() else np.nan,
                "off_power_zero_cooling_sum_c": float(cooling.sum()) if len(cooling) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _target_temperature(frame: pd.DataFrame) -> np.ndarray:
    for col in ("target_T_in", "T_set"):
        if col in frame:
            return frame[col].to_numpy(dtype=float)
    return np.full(len(frame), np.nan, dtype=float)


def analyze_oscillation(
    controller_log: pd.DataFrame,
    *,
    on_threshold_hz: float = 15.0,
    actual_on_threshold_hz: float = 1.0,
    short_cycle_seconds: float = 180.0,
    comfort_band_c: float = 0.5,
    high_freq_threshold_hz: float = 60.0,
    lookahead_seconds: float = 600.0,
) -> pd.DataFrame:
    rows = []
    for run, frame in controller_log.sort_values(["run", "elapsed_seconds"]).groupby("run", sort=False):
        elapsed = frame["elapsed_seconds"].to_numpy(dtype=float)
        if len(elapsed) == 0:
            continue
        step_seconds = float(np.median(np.diff(elapsed))) if len(elapsed) > 1 else 5.0
        hours = max(float(elapsed[-1] - elapsed[0] + step_seconds) / 3600.0, 1e-9)
        freq_target = frame["freq_target"].to_numpy(dtype=float)
        actual_freq = frame["freq"].to_numpy(dtype=float) if "freq" in frame else np.full(len(frame), np.nan)
        temp = frame["T_in"].to_numpy(dtype=float)
        target = _target_temperature(frame)

        target_on = freq_target >= float(on_threshold_hz)
        actual_on = actual_freq > float(actual_on_threshold_hz)
        high_target = freq_target >= float(high_freq_threshold_hz)
        near_target = np.isfinite(target) & (np.abs(temp - target) <= float(comfort_band_c))
        near_target_off = near_target & ~target_on
        high_start_count = int(np.logical_and(high_target[1:], ~high_target[:-1]).sum()) if len(high_target) > 1 else 0

        lookahead_count = max(1, int(np.ceil(float(lookahead_seconds) / max(step_seconds, 1e-9))))
        off_to_high_count = 0
        off_to_high_event_count = 0
        was_near_off = False
        for idx, is_near_off in enumerate(near_target_off):
            if not bool(is_near_off):
                was_near_off = False
                continue
            future_high = high_target[idx + 1 : min(len(high_target), idx + 1 + lookahead_count)]
            has_future_high = bool(future_high.any())
            if has_future_high:
                off_to_high_count += 1
                if not was_near_off:
                    off_to_high_event_count += 1
            was_near_off = True

        on_durations, off_durations = _cycle_durations(elapsed, actual_on)
        short_on = [duration for duration in on_durations if duration < float(short_cycle_seconds)]
        rows.append(
            {
                "run": run,
                "steps": int(len(frame)),
                "duration_h": float(hours),
                "near_target_steps": int(near_target.sum()),
                "near_target_off_steps": int(near_target_off.sum()),
                "near_target_off_ratio": float(near_target_off.sum() / max(int(near_target.sum()), 1)),
                "near_target_off_to_high_within_10min_count": int(off_to_high_count),
                "near_target_off_to_high_within_10min_event_count": int(off_to_high_event_count),
                "near_target_off_to_high_within_10min_ratio": float(off_to_high_count / max(int(near_target_off.sum()), 1)),
                "high_start_count": int(high_start_count),
                "high_starts_per_hour": float(high_start_count / hours),
                "short_on_cycle_count": int(len(short_on)),
                "short_on_cycle_ratio": float(len(short_on) / max(len(on_durations), 1)),
                **_duration_stats("on", on_durations),
                **_duration_stats("off", off_durations),
            }
        )
    return pd.DataFrame(rows)


def analyze_action_ood(
    controller_log: pd.DataFrame,
    checkpoint_path: Path,
    *,
    ood_z_threshold: float = 2.0,
    extreme_z_threshold: float = 3.0,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    action_cols = list(checkpoint["target_action_cols"])
    norm = checkpoint["target_action_norm"]
    mean = np.asarray(norm["mean"], dtype=float)
    std = np.maximum(np.asarray(norm["std"], dtype=float), 1e-9)
    rows = []
    for run, frame in controller_log.sort_values(["run", "elapsed_seconds"]).groupby("run", sort=False):
        actions = frame[action_cols].to_numpy(dtype=float)
        z = (actions - mean.reshape(1, -1)) / std.reshape(1, -1)
        abs_z = np.abs(z)
        row = {
            "run": run,
            "steps": int(len(frame)),
            "mean_abs_action_z": float(abs_z.mean()),
            "max_abs_action_z": float(abs_z.max()),
            "action_z_gt_threshold_ratio": float((abs_z > float(ood_z_threshold)).mean()),
            "action_z_gt_extreme_ratio": float((abs_z > float(extreme_z_threshold)).mean()),
        }
        any_ood = (abs_z > float(ood_z_threshold)).any(axis=1)
        any_extreme = (abs_z > float(extreme_z_threshold)).any(axis=1)
        row["any_action_ood_step_ratio"] = float(any_ood.mean())
        row["any_action_extreme_step_ratio"] = float(any_extreme.mean())
        for idx, col in enumerate(action_cols):
            row[f"{col}_mean_z"] = float(z[:, idx].mean())
            row[f"{col}_mean_abs_z"] = float(abs_z[:, idx].mean())
            row[f"{col}_max_abs_z"] = float(abs_z[:, idx].max())
            row[f"{col}_z_gt_threshold_ratio"] = float((abs_z[:, idx] > float(ood_z_threshold)).mean())
            row[f"{col}_z_gt_extreme_ratio"] = float((abs_z[:, idx] > float(extreme_z_threshold)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_action_stats(action_stats: pd.DataFrame) -> pd.DataFrame:
    if action_stats.empty:
        return pd.DataFrame()
    numeric = action_stats.select_dtypes(include=[np.number]).columns.tolist()
    row = action_stats[numeric].mean(numeric_only=True).to_dict()
    row["runs"] = int(len(action_stats))
    return pd.DataFrame([row])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.rollout_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    controller_log = pd.read_csv(args.rollout_dir / "controller_log.csv", encoding="utf-8-sig")
    action_stats = analyze_actions(
        controller_log,
        on_threshold_hz=args.on_threshold_hz,
        actual_on_threshold_hz=args.actual_on_threshold_hz,
        short_cycle_seconds=args.short_cycle_seconds,
        high_fan_threshold=args.high_fan_threshold,
    )
    action_stats.to_csv(output_dir / "action_cycle_statistics.csv", index=False, encoding="utf-8-sig")
    aggregate_action_stats(action_stats).to_csv(output_dir / "action_cycle_aggregate.csv", index=False, encoding="utf-8-sig")
    oscillation_stats = analyze_oscillation(
        controller_log,
        on_threshold_hz=args.on_threshold_hz,
        actual_on_threshold_hz=args.actual_on_threshold_hz,
        short_cycle_seconds=args.short_cycle_seconds,
        comfort_band_c=args.comfort_band_c,
        high_freq_threshold_hz=args.high_freq_threshold_hz,
        lookahead_seconds=args.lookahead_seconds,
    )
    oscillation_stats.to_csv(output_dir / "oscillation_statistics.csv", index=False, encoding="utf-8-sig")
    aggregate_action_stats(oscillation_stats).to_csv(output_dir / "oscillation_aggregate.csv", index=False, encoding="utf-8-sig")

    trajectory_path = args.rollout_dir / "trajectories.csv"
    if trajectory_path.exists():
        trajectories = pd.read_csv(trajectory_path, encoding="utf-8-sig")
        off_cooling = analyze_off_cooling(trajectories, actual_on_threshold_hz=args.actual_on_threshold_hz)
        off_cooling.to_csv(output_dir / "off_power_zero_cooling_statistics.csv", index=False, encoding="utf-8-sig")
        if not off_cooling.empty:
            off_cooling.mean(numeric_only=True).to_frame().T.to_csv(
                output_dir / "off_power_zero_cooling_aggregate.csv",
                index=False,
                encoding="utf-8-sig",
            )
    if args.checkpoint is not None:
        action_ood = analyze_action_ood(
            controller_log,
            args.checkpoint,
            ood_z_threshold=args.ood_z_threshold,
            extreme_z_threshold=args.extreme_z_threshold,
        )
        action_ood.to_csv(output_dir / "action_ood_statistics.csv", index=False, encoding="utf-8-sig")
        aggregate_action_stats(action_ood).to_csv(output_dir / "action_ood_aggregate.csv", index=False, encoding="utf-8-sig")
    print(action_stats.to_string(index=False))


if __name__ == "__main__":
    main()
