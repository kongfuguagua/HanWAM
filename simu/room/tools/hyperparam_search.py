"""Quick hyperparameter search for the autoregressive h=24 cooling model.

Only uses data/dataset and avoids energy-derived features, per user constraints.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from simu.room.continuous_model import (
    CONTROL_COLUMNS, DT_SECONDS, PLANT_STATE_COLUMNS,
    HeatLoadServoConfig, AutoregressiveFeatureState, FeatureConfig,
)
from simu.room.tools.data_pipeline import assign_groupwise_splits, discover_cooling_runs
from simu.room.tools.train_continuous import collect_autoregressive_samples


def evaluate_model(model_path: Path) -> dict:
    """Run closed-loop evaluation on data/dataset and return summary metrics."""
    from simu.room.simulator import ContinuousEnthalpyRoomEnv

    runs = discover_cooling_runs(Path("data/dataset"))
    assign_groupwise_splits(runs)

    per_run = []
    for run in runs:
        simulator = ContinuousEnthalpyRoomEnv(model_path)
        simulator.reset(run["frame"].iloc[0])
        controls = run["frame"][CONTROL_COLUMNS].to_numpy(np.float32)[:-1]
        temps = [simulator.temperature]
        for control in controls:
            simulator.prefix.update_temperature(simulator.temperature)
            simulator.prefix.update_control(control)
            simulator.step(*control)
            temps.append(simulator.temperature)
        predicted = np.asarray(temps)
        measured = run["frame"]["T_in"].to_numpy(dtype=float)[:len(predicted)]
        error = predicted - measured
        per_run.append({
            "run": run["name"],
            "split": run["split"],
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_abs": float(np.max(np.abs(error))),
            "shape_r": float(np.corrcoef(predicted, measured)[0, 1]),
        })

    df = pd.DataFrame(per_run)
    test = df[df["split"] == "test"]
    train = df[df["split"] == "train"]
    return {
        "test_mae": float(test["mae"].mean()),
        "test_rmse": float(test["rmse"].mean()),
        "test_max_abs": float(test["max_abs"].max()),
        "test_worst_mae": float(test["mae"].max()),
        "test_min_shape_r": float(test["shape_r"].min()),
        "train_mae": float(train["mae"].mean()),
        "train_rmse": float(train["rmse"].mean()),
    }


def train_one(
    train_runs: list[dict],
    medians: np.ndarray,
    feature_config: FeatureConfig,
    hyperparams: dict,
    output_path: Path,
) -> None:
    x, y = collect_autoregressive_samples(
        train_runs, medians, stride=1, target="absolute", horizon=24,
        feature_config=feature_config, weighted=False,
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=hyperparams["learning_rate"],
        max_iter=hyperparams["max_iter"],
        max_leaf_nodes=hyperparams["max_leaf_nodes"],
        min_samples_leaf=hyperparams["min_samples_leaf"],
        l2_regularization=hyperparams["l2_regularization"],
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=42,
    ).fit(x, y)

    payload = {
        "model": model,
        "feature_variant": "autoregressive_absolute_h24",
        "supported_mode": 1,
        "state_columns": PLANT_STATE_COLUMNS,
        "state_medians": medians,
        "control_columns": CONTROL_COLUMNS,
        "dt_seconds": DT_SECONDS,
        "heat_load_config": asdict(HeatLoadServoConfig()),
        "continuity": {"tau_seconds": 0.0, "max_rate_c_per_min": 0.5},
        "autoregressive_horizon": 24,
        "feature_config": asdict(feature_config),
        "metadata": {
            "version": "v3_continuous",
            "training_runs": len(train_runs),
            "temperature_state_equation": "T[k+1]=T[k]+clip(model_T_next[k]/H, rate-limits)",
            "hyperparams": hyperparams,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path, compress=3)


def main() -> None:
    runs = discover_cooling_runs(Path("data/dataset"))
    assign_groupwise_splits(runs)
    train = [r for r in runs if r["split"] == "train"]

    states = pd.concat([r["frame"][PLANT_STATE_COLUMNS] for r in train], ignore_index=True)
    medians = states.median().reindex(PLANT_STATE_COLUMNS).fillna(0.0).to_numpy(np.float32)

    hyperparams_grid = [
        {"learning_rate": 0.05, "max_iter": 800, "max_leaf_nodes": 31,
         "min_samples_leaf": 40, "l2_regularization": 1.0},
        {"learning_rate": 0.08, "max_iter": 500, "max_leaf_nodes": 31,
         "min_samples_leaf": 40, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 800, "max_leaf_nodes": 31,
         "min_samples_leaf": 80, "l2_regularization": 2.0},
        {"learning_rate": 0.06, "max_iter": 800, "max_leaf_nodes": 31,
         "min_samples_leaf": 24, "l2_regularization": 0.5},
        {"learning_rate": 0.06, "max_iter": 500, "max_leaf_nodes": 63,
         "min_samples_leaf": 40, "l2_regularization": 1.0},
    ]

    configs = [
        ("baseline", FeatureConfig(batch1_control_memory=False, batch2_approach_temps=False, batch3_interactions=False)),
        ("batch1", FeatureConfig(batch1_control_memory=True, batch2_approach_temps=False, batch3_interactions=False)),
    ]

    results = []
    for cfg_name, feature_config in configs:
        for hp in hyperparams_grid:
            name = f"{cfg_name}_lr{hp['learning_rate']}_leaf{hp['min_samples_leaf']}_l2{hp['l2_regularization']}_iter{hp['max_iter']}"
            output_path = Path(f"simu/room/hyperparam_search/{name}.joblib")
            print(f"Training {name}...")
            train_one(train, medians, feature_config, hp, output_path)
            metrics = evaluate_model(output_path)
            metrics["name"] = name
            metrics["config"] = cfg_name
            metrics["hyperparams"] = json.dumps(hp)
            results.append(metrics)
            print(f"  test_mae={metrics['test_mae']:.4f} worst={metrics['test_worst_mae']:.4f} min_r={metrics['test_min_shape_r']:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_mae")
    print("\nTop by test MAE:")
    print(results_df[["name", "test_mae", "test_worst_mae", "test_min_shape_r", "train_mae"]].head(10).to_string(index=False))
    print("\nTop by worst-run MAE:")
    print(results_df[["name", "test_mae", "test_worst_mae", "test_min_shape_r", "train_mae"]].sort_values("test_worst_mae").head(10).to_string(index=False))

    results_df.to_csv("simu/room/hyperparam_search/results.csv", index=False, encoding="utf-8-sig")
    print("Saved simu/room/hyperparam_search/results.csv")


if __name__ == "__main__":
    main()
