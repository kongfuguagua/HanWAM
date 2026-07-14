"""Small usage example for the hybrid room simulator."""
from __future__ import annotations

from pathlib import Path

from simu.room_hybrid import HybridRoomEnv
from simu.room_hybrid.api import ACTION_COLUMNS
from simu.room_hybrid.data import load_status_csv


def main() -> None:
    csv_path = Path("data/dataset_own/1_data_202607120204.csv")
    frame = load_status_csv(csv_path)
    env = HybridRoomEnv()
    actions = frame[ACTION_COLUMNS].to_numpy("float32")[:-1]
    result = env.simulate(frame.iloc[0], actions)
    print(result[["elapsed_seconds", "T_in", "T_in_slow", "T_in_residual"]].tail())


if __name__ == "__main__":
    main()
