"""Train the self-contained V3 continuous cooling-room plant model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..continuous_model import (
    CONTROL_COLUMNS, DT_SECONDS, PLANT_STATE_COLUMNS, STATE_COLUMNS,
    HeatLoadServoConfig, AutoregressiveFeatureState, ContinuousFeatureState,
    FeatureConfig,
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
    """Legacy thermal-inertia samples (uses T_set, kept for comparison only)."""
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


def compute_sample_weights(
    frame: pd.DataFrame,
    residuals: np.ndarray | None = None,
    freq_diff_weight: float = 1.0,
    mode_change_weight: float = 2.0,
    residual_weight: float = 0.0,
) -> np.ndarray:
    """Compute per-sample weights emphasizing transitions.

    Weights are normalized to mean 1.0 so the overall loss scale is preserved.
    The returned array length matches ``len(frame)``.
    """
    freq = frame["compressor_frequency"].to_numpy(np.float64)
    mode = frame["mode"].to_numpy(np.float64)
    n = len(frame)
    weights = np.ones(n, dtype=np.float64)

    if freq_diff_weight > 0:
        freq_diff = np.abs(np.diff(freq))
        # Pad so length matches n; last sample has no successor diff.
        freq_diff = np.concatenate([freq_diff, [0.0]])
        # Normalize by a typical range (~30 Hz) to keep weights moderate.
        weights += freq_diff_weight * (freq_diff / 30.0)

    if mode_change_weight > 0:
        mode_change = np.abs(np.diff(mode)) > 0.5
        mode_change = np.concatenate([mode_change, [False]])
        weights += mode_change_weight * mode_change.astype(np.float64)

    if residual_weight > 0 and residuals is not None and len(residuals) == n:
        # Up-weight large previous residuals; clip to avoid outliers dominating.
        residual_scale = np.clip(np.abs(residuals), 0.0, 2.0)
        weights += residual_weight * residual_scale

    weights /= max(np.mean(weights), 1e-9)
    return weights.astype(np.float32)


def collect_autoregressive_samples(
    runs: list[dict],
    medians: np.ndarray,
    stride: int,
    target: str,
    horizon: int,
    feature_config: FeatureConfig,
    weighted: bool = False,
    freq_diff_weight: float = 1.0,
    mode_change_weight: float = 2.0,
    residual_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect h-step-ahead plant-model samples.

    The feature builder uses only actual plant state (no ``T_set`` or other
    controller targets).  At each step we update the current simulated
    ``T_in`` to the observed ``T_in[t]``, append the control ``controls[t]``,
    and predict either the h-step-ahead absolute temperature
    (``target="absolute"``) or the change ``ΔT = T_in[t+h] - T_in[t]``
    (``target="delta"``).

    When ``weighted=True``, sample weights are returned as a third array.
    """
    features, targets, weights = [], [], []
    for run in runs:
        frame = run["frame"]
        controls = frame[CONTROL_COLUMNS].to_numpy(np.float32)
        truth = frame["T_in"].to_numpy(np.float32)
        state = frame.loc[0, PLANT_STATE_COLUMNS].to_numpy(np.float32)
        state = np.where(np.isfinite(state), state, medians).astype(np.float32)
        prefix = AutoregressiveFeatureState(state, feature_config)
        max_t = len(frame) - horizon - 1

        run_residuals = None
        if residual_weight > 0:
            # Use a cheap baseline residual: deviation from a local mean trend.
            local_mean = pd.Series(truth).rolling(window=24, min_periods=1,
                                                   center=True).mean().to_numpy()
            run_residuals = np.abs(truth - local_mean)

        for t in range(max_t + 1):
            prefix.update_temperature(truth[t])
            prefix.update_control(controls[t])
            if t % stride == 0:
                features.append(prefix.feature())
                if target == "absolute":
                    targets.append(truth[t + horizon])
                else:
                    targets.append(truth[t + horizon] - truth[t])

        if weighted:
            run_weights = compute_sample_weights(
                frame.iloc[:max_t + 1],
                residuals=run_residuals[:max_t + 1] if run_residuals is not None else None,
                freq_diff_weight=freq_diff_weight,
                mode_change_weight=mode_change_weight,
                residual_weight=residual_weight,
            )
            # Subsample weights with the same stride used for features.
            weights.extend(run_weights[::stride])

    features = np.asarray(features, np.float32)
    targets = np.asarray(targets, np.float32)
    if weighted:
        return features, targets, np.asarray(weights, np.float32)
    return features, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path,
                        default=Path("simu/room/continuous_cooling_model.joblib"))
    parser.add_argument("--variant", type=str, default="autoregressive",
                        choices=["thermal_inertia", "autoregressive"],
                        help="Feature/target formulation to train.")
    parser.add_argument("--target", type=str, default="absolute",
                        choices=["delta", "absolute"],
                        help="Autoregressive target: predict next T_in or ΔT.")
    parser.add_argument("--horizon", type=int, default=24,
                        help="Autoregressive prediction horizon in 5 s steps.")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--loss", type=str, default="squared_error",
                        choices=["squared_error", "absolute_error", "quantile"],
                        help="Loss function for HistGradientBoostingRegressor.")
    parser.add_argument("--quantile-alpha", type=float, default=0.5,
                        help="Alpha for quantile loss.")

    # Feature-batch ablation flags.
    parser.add_argument("--no-batch1", action="store_false", dest="batch1",
                        help="Disable batch 1: control memory features.")
    parser.add_argument("--no-batch2", action="store_false", dest="batch2",
                        help="Disable batch 2: approach-temperature EWM features.")
    parser.add_argument("--no-batch3", action="store_false", dest="batch3",
                        help="Disable batch 3: interaction features.")
    parser.set_defaults(batch1=True, batch2=True, batch3=True)

    # Weighted training flags.
    parser.add_argument("--weighted", action="store_true",
                        help="Use transition-emphasized sample weights.")
    parser.add_argument("--freq-diff-weight", type=float, default=1.0)
    parser.add_argument("--mode-change-weight", type=float, default=2.0)
    parser.add_argument("--residual-weight", type=float, default=0.0)

    args = parser.parse_args()

    if args.stride is None:
        args.stride = 1 if args.variant == "autoregressive" else 180

    runs = discover_cooling_runs(args.data_dir)
    assign_groupwise_splits(runs)
    train = [run for run in runs if run["split"] == "train"]

    if args.variant == "autoregressive":
        state_cols = PLANT_STATE_COLUMNS
    else:
        state_cols = STATE_COLUMNS

    states = pd.concat([run["frame"][state_cols] for run in train], ignore_index=True)
    medians = states.median().reindex(state_cols).fillna(0.0).to_numpy(np.float32)

    feature_config = FeatureConfig(
        batch1_control_memory=args.batch1,
        batch2_approach_temps=args.batch2,
        batch3_interactions=args.batch3,
    )

    if args.variant == "autoregressive":
        result = collect_autoregressive_samples(
            train, medians, args.stride, args.target, args.horizon,
            feature_config, weighted=args.weighted,
            freq_diff_weight=args.freq_diff_weight,
            mode_change_weight=args.mode_change_weight,
            residual_weight=args.residual_weight,
        )
        if args.weighted:
            x, y, sample_weight = result
        else:
            x, y = result
            sample_weight = None
    else:
        x, y = collect_samples(train, medians, args.stride)
        sample_weight = None

    model_kwargs = dict(
        loss=args.loss, learning_rate=0.06, max_iter=args.max_iter,
        max_leaf_nodes=31, min_samples_leaf=24, l2_regularization=0.5,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=25,
        random_state=42,
    )
    if args.loss == "quantile":
        model_kwargs["quantile"] = args.quantile_alpha
    if args.weighted:
        # Slightly stronger regularization because weighted samples reduce
        # effective sample size.
        model_kwargs.update(min_samples_leaf=40, l2_regularization=1.0)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model = HistGradientBoostingRegressor(**model_kwargs).fit(x, y, **fit_kwargs)

    if args.variant == "autoregressive":
        continuity = {"tau_seconds": 0.0, "max_rate_c_per_min": 0.5}
        target_label = "T_next" if args.target == "absolute" else "delta_T"
        state_equation = f"T[k+1]=T[k]+clip(model_{target_label}[k]/H, rate-limits)"
        feature_variant = f"autoregressive_{args.target}_h{args.horizon}"
    else:
        continuity = {"tau_seconds": 90.0, "max_rate_c_per_min": 0.5}
        state_equation = "T[k+1]=T[k]+alpha*(T_direct[k+1]-T[k])"
        feature_variant = "thermal_inertia"

    payload = {
        "model": model, "feature_variant": feature_variant, "supported_mode": 1,
        "state_columns": state_cols, "state_medians": medians,
        "control_columns": CONTROL_COLUMNS, "dt_seconds": DT_SECONDS,
        "heat_load_config": asdict(HeatLoadServoConfig()),
        "continuity": continuity,
        "autoregressive_horizon": args.horizon if args.variant == "autoregressive" else None,
        "feature_config": asdict(feature_config),
        "metadata": {
            "version": "v3_continuous", "training_runs": len(train),
            "temperature_state_equation": state_equation,
            "weighted": args.weighted,
            "loss": args.loss,
            "freq_diff_weight": args.freq_diff_weight if args.weighted else None,
            "mode_change_weight": args.mode_change_weight if args.weighted else None,
            "residual_weight": args.residual_weight if args.weighted else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output, compress=3)
    print(f"train_runs={len(train)} samples={len(y)} features={x.shape[1]}")
    if sample_weight is not None:
        print(f"sample_weight mean={np.mean(sample_weight):.3f} max={np.max(sample_weight):.3f}")
    print(args.output)


if __name__ == "__main__":
    main()
