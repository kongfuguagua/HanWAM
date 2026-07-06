"""Train the self-contained V3 continuous cooling-room model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..continuous_model import (
    CONTROL_COLUMNS, DT_SECONDS, STATE_COLUMNS, HeatLoadServoConfig,
    ContinuousFeatureState,
)
from .data_pipeline import assign_groupwise_splits, discover_cooling_runs


def selected_horizons(remaining: int) -> list[int]:
    dense = list(range(1, 13))
    sparse = [18, 24, 36, 48, 60, 90, 120, 180, 240, 360, 480,
              720, 960, 1200, 1440, 1800, 2160, 2520, 2880]
    values = [h for h in dense + sparse if h <= remaining]
    if remaining > 0 and remaining not in values:
        values.append(remaining)
    return sorted(set(values))


def collect_samples(runs: list[dict], medians: np.ndarray, stride: int):
    features, targets = [], []
    for run in runs:
        frame = run["frame"]
        controls = frame[CONTROL_COLUMNS].to_numpy(np.float32)
        truth = frame["T_in"].to_numpy(np.float32)
        for start in range(0, max(1, len(frame) - 12), stride):
            horizons = selected_horizons(len(frame) - start - 1)
            if not horizons:
                continue
            state = frame.loc[start, STATE_COLUMNS].to_numpy(np.float32)
            state = np.where(np.isfinite(state), state, medians).astype(np.float32)
            prefix = ContinuousFeatureState(state)
            wanted = set(horizons)
            for horizon in range(1, max(horizons) + 1):
                prefix.update(controls[start + horizon - 1])
                if horizon in wanted:
                    features.append(prefix.feature())
                    targets.append(truth[start + horizon] - float(frame["T_set"].iloc[start]))
    return np.asarray(features, np.float32), np.asarray(targets, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path,
                        default=Path("simu/room/continuous_cooling_model.joblib"))
    parser.add_argument("--stride", type=int, default=180)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    runs = discover_cooling_runs(args.data_dir)
    assign_groupwise_splits(runs)
    train = [run for run in runs if run["split"] == "train"]
    states = pd.concat([run["frame"][STATE_COLUMNS] for run in train], ignore_index=True)
    medians = states.median().reindex(STATE_COLUMNS).fillna(0.0).to_numpy(np.float32)
    x, y = collect_samples(train, medians, args.stride)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=args.max_iter,
        max_leaf_nodes=31, min_samples_leaf=24, l2_regularization=0.5,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=25,
        random_state=42,
    ).fit(x, y)

    payload = {
        "model": model, "feature_variant": "thermal_inertia", "supported_mode": 1,
        "state_columns": STATE_COLUMNS, "state_medians": medians,
        "control_columns": CONTROL_COLUMNS, "dt_seconds": DT_SECONDS,
        "heat_load_config": asdict(HeatLoadServoConfig()),
        "continuity": {"tau_seconds": 90.0, "max_rate_c_per_min": 0.5},
        "metadata": {
            "version": "v3_continuous", "training_runs": len(train),
            "temperature_state_equation": "T[k+1]=T[k]+alpha*(T_direct[k+1]-T[k])",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output, compress=3)
    print(f"train_runs={len(train)} samples={len(y)} features={x.shape[1]}")
    print(args.output)


if __name__ == "__main__":
    main()
