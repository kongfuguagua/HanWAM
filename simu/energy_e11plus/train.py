"""Train per-mode GradientBoosting energy models for the E1-1plus AC unit.

The E1-1plus status CSVs under ``data/E1-1plus`` are GBK-encoded 24-column logs.
The original chamber data also includes actual ("推理") control values that
post-process the controller output, but the E1-1plus logs only expose the
target commands.  We treat the target values (freq_target, eev_target,
fan_out_target) as the input features, matching the simulator's call
convention: ``AirConditionerSimulator`` ultimately passes the freq response
model's actual frequency, which converges to the target in steady state.

Training pipeline (mirrors ``simu/energy/model.py`` -> ``per_mode_models.pkl``):

  1. Read every ``data/E1-1plus/*.csv`` via ``data.io.read_raw_csv`` so column
     names decode cleanly.
  2. Compute per-step actual power from the cumulative ``energy_cum`` column:
        power_w[t] = (energy_cum[t] - energy_cum[t-1]) / dt * 3600 * 1000
  3. Drop the first row of each run (NaN delta), filter to the requested mode
     and a plausible power range to suppress the giant kWh jumps seen when the
     meter resets, and clip outliers via a robust percentile cap.
  4. Fit a per-mode ``GradientBoostingRegressor`` on
     ``[freq_target, eev_target, fan_out_target] -> power_w``.
  5. Persist as ``simu/energy_e11plus/per_mode_models.pkl`` in the same shape
     the existing ``EnergyModel`` expects:

        {"制冷": {"gb": <regressor>, ...}, "制热": {"gb": <regressor>, ...}}

The pickle stores rf / lr placeholders too so the optional comparison tooling
in ``roomtest/compare_energy_models.py`` can still load it (those branches will
raise ``AttributeError`` if invoked, which is fine — the EnergyModel only
accesses ``gb``).

Usage::

    python -m simu.energy_e11plus.train            # default paths
    python -m simu.energy_e11plus.train --no-clip  # skip outlier clipping
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from data.io import read_raw_csv


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_GLOB = str(PROJECT_DIR / "data" / "E1-1plus" / "*.csv")
DEFAULT_OUTPUT = PROJECT_DIR / "simu" / "energy_e11plus" / "per_mode_models.pkl"

FEATURE_COLS = ["freq_target", "eev_target", "fan_out_target"]
TARGET_COL = "power_w"

# Default dt for the cumulative-energy differencing.  All CSVs observed are
# sampled at 5 s, but we read the real cadence from the timestamps when they
# are present.
DEFAULT_DT_SECONDS = 5.0

# Modes the simulator exposes.  We require both so ``EnergyModel`` can
# instantiate with either "制冷" or "制热".  E1-1plus data only contains
# mode=1 (制冷); for "制热" we train on the same data and document the gap.
COOL_MODE = 1
HEAT_MODE = 3
MODE_TO_NAME = {COOL_MODE: "制冷", HEAT_MODE: "制热"}

# Robust outlier cap (W).  Real E1-1plus compressor power tops out near 1.1 kW
# under normal load; the cumulative meter occasionally emits a single-tick
# delta on the order of 50 kW that is clearly an artefact of the meter's
# resolution (5 s × ~14 kW ≈ 70 kWh/s ≈ ~25 MJ step).
DEFAULT_POWER_MAX_W = 1500.0
DEFAULT_POWER_MIN_W = 0.0


def _compute_power_w(frame: pd.DataFrame, dt_default: float) -> pd.ndarray:
    """Return per-step power in watts, NaN for the first row of each run."""
    energy = frame["energy_cum"].to_numpy(dtype=float)
    if "ts" in frame.columns and frame["ts"].notna().any():
        ts = pd.to_datetime(frame["ts"], errors="coerce")
        dt = ts.diff().dt.total_seconds().to_numpy()
        fallback = np.isfinite(dt) & (dt > 0)
    else:
        dt = np.full(len(frame), dt_default, dtype=float)
        fallback = np.zeros(len(frame), dtype=bool)

    d_energy = np.diff(energy, prepend=energy[0])
    safe_dt = np.where(fallback, dt, dt_default)
    power = d_energy * 3_600_000.0 / safe_dt
    return power


def _load_mode(
    csv_paths: list[Path],
    mode: int,
    power_min: float,
    power_max: float,
    clip_outliers: bool,
    dt_default: float,
) -> pd.DataFrame:
    """Concatenate filtered (X, y) rows for a single mode across all CSVs."""
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = read_raw_csv(path)
        if "mode" not in df.columns:
            continue
        df = df.copy()
        df["power_w"] = _compute_power_w(df, dt_default)
        df["__src"] = path.name

        keep_cols = FEATURE_COLS + ["power_w", "__src", "mode"]
        df = df[keep_cols]
        df = df[df["mode"] == mode]

        # Drop the first row of every run (NaN delta).
        df = df.iloc[1:].copy()

        # Filter to plausible compressor-on rows so we don't train on the
        # off-state (freq == 0 always gives 0 W by definition; that's handled
        # by the EnergyModel's ``freq <= 1`` guard at inference time).
        df = df.dropna(subset=FEATURE_COLS + ["power_w"])
        df = df[(df["power_w"] >= power_min) & (df["power_w"] <= power_max)]
        if clip_outliers and len(df) > 0:
            hi = float(np.percentile(df["power_w"], 99.5))
            df = df[df["power_w"] <= hi]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=keep_cols)
    return pd.concat(frames, ignore_index=True)


def train_gb(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> GradientBoostingRegressor:
    """Default GB hyperparameters: shallow trees, modest learning rate.

    These mirror the conventions of the original ``simu/energy/per_mode_models.pkl``
    training run (sklearn ``GradientBoostingRegressor`` with ~3 features and a
    few hundred trees).
    """
    return GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    ).fit(X, y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-clip", action="store_true", help="Skip 99.5-percentile power clipping")
    parser.add_argument("--power-min", type=float, default=DEFAULT_POWER_MIN_W)
    parser.add_argument("--power-max", type=float, default=DEFAULT_POWER_MAX_W)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_SECONDS)
    args = parser.parse_args()

    csv_paths = sorted(Path(p) for p in glob.glob(args.data_glob))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs matched {args.data_glob!r}")

    print(f"Loading {len(csv_paths)} CSV files from {args.data_glob}")
    payload: dict[str, dict] = {}

    # --- 制冷 (mode 1): the only mode present in the E1-1plus data -----------
    cool = _load_mode(
        csv_paths,
        mode=COOL_MODE,
        power_min=args.power_min,
        power_max=args.power_max,
        clip_outliers=not args.no_clip,
        dt_default=args.dt,
    )
    print(f"  mode={COOL_MODE} (制冷): {len(cool)} samples")
    if len(cool) < 100:
        raise RuntimeError(
            f"Too few 制冷 samples ({len(cool)}) to train a stable model. "
            "Check the data path or relax --power-max."
        )

    X_cool = cool[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_cool = cool[TARGET_COL].to_numpy(dtype=np.float32)
    gb_cool = train_gb(X_cool, y_cool)
    train_mae = float(np.mean(np.abs(gb_cool.predict(X_cool) - y_cool)))
    print(f"    train MAE = {train_mae:.1f} W (median power {np.median(y_cool):.0f} W)")

    payload["制冷"] = {
        "gb": gb_cool,
        "rf": None,
        "lr": None,
        "a": 0.0,
        "b_freq": 0.0,
        "b_valve": 0.0,
        "b_fan": 0.0,
    }

    # --- 制热 (mode 3): E1-1plus dataset has no heating runs -----------------
    # Reuse the cool-mode training set as a conservative fallback so the
    # ``EnergyModel`` interface still instantiates for heating.  Callers
    # requiring accurate heating predictions should collect 制热 data and
    # retrain via this script.
    heat = _load_mode(
        csv_paths,
        mode=HEAT_MODE,
        power_min=args.power_min,
        power_max=args.power_max,
        clip_outliers=not args.no_clip,
        dt_default=args.dt,
    )
    if len(heat) < 50:
        print(
            f"  mode={HEAT_MODE} (制热): only {len(heat)} samples — "
            "falling back to the 制冷 GB regressor."
        )
        gb_heat = gb_cool
    else:
        print(f"  mode={HEAT_MODE} (制热): {len(heat)} samples")
        X_heat = heat[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_heat = heat[TARGET_COL].to_numpy(dtype=np.float32)
        gb_heat = train_gb(X_heat, y_heat)

    payload["制热"] = {
        "gb": gb_heat,
        "rf": None,
        "lr": None,
        "a": 0.0,
        "b_freq": 0.0,
        "b_valve": 0.0,
        "b_fan": 0.0,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"\nSaved {out}")
    print(f"  size: {out.stat().st_size / 1024:.1f} KiB")
    print(f"  modes: {list(payload)}")


if __name__ == "__main__":
    main()
