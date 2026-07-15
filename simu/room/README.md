# Continuous cooling-room simulator V3

`simu.room` contains the V3 continuous cooling-room environment:

```text
continuous_model.py  online features and heat-load diagnostic state
simulator.py         reset/step/simulate environment
example_usage.py     minimal online example
tools/               data, training, and evaluation scripts
```

The default local model path is `continuous_cooling_model.joblib`, but the
model artifact is not committed in this clean repository. Tests create a
temporary mock joblib payload when simulator behavior needs coverage.

## Minimal Interface

```python
from simu.room.simulator import ContinuousEnthalpyRoomEnv

env = ContinuousEnthalpyRoomEnv("simu/room/continuous_cooling_model.joblib")
env.reset(T_out=35.0, T_in=30.0, T_out_coil=37.0, T_in_coil=24.0)

for _ in range(720):
    observation = env.step(freq=40.0, eev=180.0, fan_out=750.0)

print(observation["T_in"])
```

V3 treats direct model output as a target and integrates a causal thermal state
with a bounded temperature rate. Compressor-off operation can use a passive
outdoor heat-load fallback so fan-only trajectories warm physically.

V4 lives separately under `simu/roomv4`.
