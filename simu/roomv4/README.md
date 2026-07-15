# Hybrid room simulator V4

V4 is an experimental hybrid room model line independent from V3. It keeps the
runtime interface focused on physical observations plus three controls:

```text
simulator.py            reset/step/simulate online environment
following_simulator.py  V4.1 following environment
model.py                V4 dynamic model
data.py                 dataset discovery and split helpers
train.py / evaluate.py  training and evaluation scripts
plot_curves.py          plotting script
example_usage.py        minimal online example
```

The default model path is `room_v4_model.pt`, but trained Torch artifacts and
generated output folders are not committed here.

## Minimal Interface

```python
from simu.roomv4.simulator import HybridRoomV4Env

env = HybridRoomV4Env("simu/roomv4/room_v4_model.pt")
env.reset(T_out=35.0, T_in=30.0, T_out_coil=37.0, T_in_coil=24.0)
observation = env.step(freq=40.0, eev=180.0, fan_out=750.0)
```

For faster local tracking experiments:

```python
from simu.roomv4.following_simulator import HybridRoomV4FollowingEnv

env = HybridRoomV4FollowingEnv(fast_weight=0.5)
env.reset(initial_observation=frame.iloc[0])
observation = env.step(freq=40.0, eev=180.0, fan_out=750.0)
```

Training and evaluation expect an external dataset root:

```bash
python -m simu.roomv4.train --data-dir data/dataset_full
python -m simu.roomv4.evaluate --data-dir data/dataset_full
```
